# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import os
import random
import time
from distutils.util import strtobool
from  rlmodel.controlnet import StyleExpert,Expert
import gym 
from envs import StyledLunarLander,LunarLander_Factors,StyledCartPoleEnv,CartPole_Factors
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils import get_style_state

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--device", type=str,default='cuda:3')
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="my-test-project",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default='jingtianji',
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env_name", type=str, default="CartPole",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=2000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=16,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=512,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--minibatch_size", type=int, default=512,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_handle, seed, idx, capture_video, run_name):
    def thunk():
        env = env_handle()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def make_styled_env(env_handle,seed,reward_factors,idx = 0, capture_video = False, run_name = "Test"):
    def thunk():
        env = env_handle(reward_factors = reward_factors)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

class Learner(object):
    def __init__(self,args,env_handle,factors:dict,allow_retrain) -> None:
        self.args = args 
        self.env_handle = env_handle
        env = env_handle()
        self.obs_dim,self.act_dim = env.observation_space.shape[0],env.action_space.n
        self.factors = factors
        self.style_dim = len(self.factors.keys())
        self.agent = StyleExpert(self.obs_dim,self.act_dim,self.style_dim,parent=None,allow_retrain=allow_retrain)
        self.pi_optimizer = optim.Adam(self.agent.style_actor.parameters(), lr=self.args.learning_rate, eps=1e-5)
        self.v_optimizer = optim.Adam(self.agent.style_critic.parameters(), lr=self.args.learning_rate, eps=1e-5)
        self.device = torch.device(args.device)
        self.agent.to(self.device)
        self.num_envs = self.args.num_envs
        self.reset_envs()
        self._set_seed(args.seed)
        self.global_step = 0
        self.start_time = time.time()
    
    def generate_reward_factors(self,seed = 0):
        random.seed(seed)
        reward_factors = {}
        style_state = []
        for factor_name, factor_info in self.factors.items():
            if factor_info['fine_grit'] > 0:
                if factor_info['lower_bound'] == factor_info['upper_bound']:
                    reward_factors[factor_name] = factor_info['lower_bound']
                else:
                    grit_factor = np.linspace(
                        factor_info['lower_bound'],
                        factor_info['upper_bound'],
                        int((factor_info['upper_bound'] - factor_info['lower_bound']) / factor_info['fine_grit']) + 1
                    )
                    reward_factors[factor_name] = random.choice(grit_factor)
            else:
                reward_factors[factor_name] = \
                    random.uniform(factor_info['lower_bound'], factor_info['upper_bound'])

            if factor_info['lower_bound'] == factor_info['upper_bound']:
                style_value= factor_info['upper_bound'] / 2
            else:
                style_value = (reward_factors[factor_name] - factor_info['lower_bound']) / \
                                (factor_info['upper_bound'] - factor_info['lower_bound'])
                
            style_state.append(style_value)
        return reward_factors,style_state

    def reset_envs(self):
        env_list = []
        self.reward_factors = []
        self.style_states = []
        for i in range(self.num_envs):
            reward_factor,style_state = self.generate_reward_factors(self.args.seed + i)
            env_list.append(make_styled_env(self.env_handle,self.args.seed+i,reward_factor))
            self.reward_factors.append(reward_factor)
            self.style_states.append(np.array(style_state))
        self.envs = gym.vector.SyncVectorEnv(env_list)
        ## transform the style states into tensor
        self.style_states = np.stack(self.style_states,axis=0)
        self.style_states = torch.from_numpy(self.style_states).float().to(self.device)

        ## reset the envs 
        self.next_obs = torch.Tensor(self.envs.reset()).to(self.device)
        self.next_done = torch.zeros(self.num_envs).to(self.device)

    def learn(self,total_timesteps,num_steps,reset_env_freq = 5):
        self._init_track()
        batch_size = num_steps * self.num_envs
        num_updates = total_timesteps // batch_size
        
        self.global_step = 0
        self.start_time = time.time()
        for update in range(1,num_updates + 1):
            if self.args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.args.learning_rate
                self.pi_optimizer.param_groups[0]["lr"] = lrnow
                self.v_optimizer.param_groups[0]["lr"] = lrnow
            
            if update % reset_env_freq == 0:
                self.reset_envs() ## reset the reward factors and style states 
            rollout_results = self.rollout(num_steps=num_steps,num_envs=self.num_envs,device=self.device)
            self.update(**rollout_results)

    def rollout(self,num_steps,num_envs,device):
        args = self.args
        obs = torch.zeros((num_steps, num_envs) + (self.obs_dim,)).to(device)
        styles = torch.zeros((num_steps,num_envs) + (self.style_dim,)).to(device)
        actions = torch.zeros((num_steps, num_envs) + self.envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((num_steps, num_envs)).to(device)
        rewards = torch.zeros((num_steps, num_envs)).to(device)
        dones = torch.zeros((num_steps, num_envs)).to(device)
        values = torch.zeros((num_steps, num_envs)).to(device)

        # TRY NOT TO MODIFY: start the game
        for step in range(0, num_steps):
            self.global_step += 1 * num_envs
            obs[step] = self.next_obs
            styles[step] = self.style_states
            dones[step] = self.next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(self.next_obs,self.style_states)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = self.envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            self.next_obs, self.next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    # print(f"global_step={self.global_step}, episodic_return={item['episode']['r']}")
                    self.writer.add_scalar("charts/episodic_return", item["episode"]["r"], self.global_step)
                    self.writer.add_scalar("charts/episodic_length", item["episode"]["l"], self.global_step)
                    reward_record = self.envs.envs[0].reward_record
                    for k,v in reward_record.items():
                        self.writer.add_scalar(f"charts/{k}",v,self.global_step)
                    break

        # bootstrap value if not done 
        with torch.no_grad():
            next_value = self.agent.get_value(self.next_obs,self.style_states).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - self.next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        rollout_result = {
            'obs':obs,'styles':styles,'actions':actions,'logprobs':logprobs,'values':values,'advantages':advantages,'returns':returns
        }
        return rollout_result
    
    def update(self,obs,styles,actions,logprobs,values,advantages,returns):
        # flatten the batch
        args = self.args
        b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
        b_styles = styles.reshape((-1,) + (self.style_dim,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        batch_size = values.shape[0]
        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds],b_styles[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                self.pi_optimizer.zero_grad()
                self.v_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.style_critic.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(self.agent.style_actor.parameters(), args.max_grad_norm)
                self.pi_optimizer.step()
                self.v_optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        self.writer.add_scalar("charts/learning_rate",self. pi_optimizer.param_groups[0]["lr"], self.global_step)
        self.writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.global_step)
        self.writer.add_scalar("losses/explained_variance", explained_var, self.global_step)
        print("SPS:", int(self.global_step / (time.time() - self.start_time)))
        self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - self.start_time)), self.global_step)

    def _init_track(self):
        run_name = f"{self.args.env_name}__{self.args.exp_name}__{self.args.seed}__{int(time.time())}"
        if self.args.track:
            import wandb

            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                sync_tensorboard=True,
                config=vars(self.args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        self.writer = SummaryWriter(f"runs/{run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
        )

    def _set_seed(self,seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

    def save(self, save_path, save_name):
        self.agent.save(save_path,save_name)
    def load(self, load_path, load_name):
        self.agent.load(load_path,load_name)



tmp_CartPole_Factors = {
    'side_factor': {"lower_bound": 0, "upper_bound": 2, "fine_grit": 0.5}
}

if __name__ == "__main__":
    args = parse_args()
    env_handle = StyledCartPoleEnv
    style_leaner = Learner(args, env_handle,tmp_CartPole_Factors,allow_retrain=False)
    style_leaner.agent.load_expert("logs/expert","CartPole")
    style_leaner.learn(args.total_timesteps,args.num_steps)
    style_leaner.save("logs/mas/cartpole","from_expert")
    

    