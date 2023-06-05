import random
import time
from  rlmodel.continuous_action import ContinuousExpert,ContinuousStyleExpert
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import mo_gymnasium as mo_gym
from algorithm.utils import make_styled_env,generate_reward_factors

class Continuous_Learner(object):
    def __init__(self,args,env_id,factors:dict) -> None:
        self.args = args 
        self.env_id = env_id
        env = mo_gym.make(env_id)
        assert isinstance(env.action_space,gym.spaces.Box)
        self.obs_dim,self.act_dim = env.observation_space.shape[0],np.prod(env.action_space.shape)
        self.factors = factors #! dictionary, predefine the bounds of each factor
        self.style_dim = len(self.factors.keys())
        if self.args.use_mas:
            print("################### Using MAS ###################")
            self.agent = ContinuousStyleExpert(self.obs_dim,self.act_dim,self.style_dim,allow_retrain=self.args.allow_retrain)
            # self.agent = MLPStyleExpert(self.obs_dim,self.act_dim,self.style_dim)
        else:
            print("################### Using Expert ###################")
            self.agent = ContinuousExpert(self.obs_dim,self.act_dim)
        self.use_mas = self.args.use_mas
        self.pi_optimizer = optim.Adam(self.agent.actor.parameters(), lr=self.args.learning_rate, eps=1e-5)
        self.v_optimizer = optim.Adam(self.agent.critic.parameters(), lr=self.args.learning_rate, eps=1e-5)
        self.device = torch.device(args.device)
        self.agent.to(self.device)
        self.num_envs = self.args.num_envs
        self.reset_envs()
        self._set_seed(args.seed)
        self.global_step = 0
        self.start_time = time.time()

    def reset_envs(self):
        env_list = []
        self.reward_factors = []
        self.style_states = []
        for i in range(self.num_envs):
            reward_factor_list,reward_factor,style_state = generate_reward_factors(self.factors,self.args.seed + i)
            env_list.append(make_styled_env(self.env_id,self.args.seed+i,reward_factor_list))
            self.reward_factors.append(reward_factor)
            self.style_states.append(np.array(style_state))
        # self.envs = mo_gym.MOSyncVectorEnv(env_list)
        # self.envs = mo_gym.MORecordEpisodeStatistics(self.envs)
        self.envs =  gym.vector.SyncVectorEnv(env_list)
        self.envs = gym.wrappers.RecordEpisodeStatistics(self.envs)
        ## transform the style states into tensor
        self.style_states = np.stack(self.style_states,axis=0)
        self.style_states = torch.from_numpy(self.style_states).float().to(self.device)

        ## reset the envs
        self.next_obs,info = self.envs.reset() 
        self.next_obs = torch.Tensor(self.next_obs).to(self.device)
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
            next_obs, reward, done,truncted, info = self.envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            self.next_obs, self.next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            if "episode" in info.keys():
                for i,terminate in enumerate(info['_episode']):
                    if terminate:
                        self.writer.add_scalar("charts/episodic_return", info['episode']['r'][i], self.global_step)
                        self.writer.add_scalar("charts/episodic_length", info['episode']['l'][i], self.global_step)
                    

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
                if epoch % args.pi_update_frq == 0:
                    self.pi_optimizer.zero_grad()
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    entropy_loss = entropy.mean()
                    pg_loss = pg_loss - args.ent_coef * entropy_loss
                    pg_loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.actor.parameters(), args.max_grad_norm)
                    self.pi_optimizer.step()

                # Value loss
                self.v_optimizer.zero_grad()
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

                v_loss = v_loss * args.vf_coef
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.agent.critic.parameters(), args.max_grad_norm)
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
        # print("SPS:", int(self.global_step / (time.time() - self.start_time)))
        self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - self.start_time)), self.global_step)

    def _init_track(self):
        run_name = f"{self.env_id}__{self.args.exp_name}"
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
        self.writer = SummaryWriter(f"{self.args.save_path}/{run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
        )

    def _set_seed(self,seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

    def save(self, save_path = None):
        if save_path is None:
            self.agent.save(self.args.model_path)
        else:
            self.agent.save(save_path)
    def load(self, load_path = None):
        if load_path is None:
            self.agent.load(self.args.model_path)
        else:
            self.agent.load(load_path)





    