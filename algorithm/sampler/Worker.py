import numpy as np
import torch as th
import ray
import os
import mo_gymnasium as mo_gym
import io 
from algorithm.utils import make_styled_env,generate_reward_factors
from algorithm.sampler.ppo_sampler import PPO_Agent
from algorithm.sampler.gae import GAE 
from algorithm.sampler.buffer import Instance
import ray 

class Worker(object):
    def __init__(self,
                 configs,
                 agent_type:PPO_Agent,
                 seed:int) -> None:
        self.env_id = configs['env_id']    
        self.env = mo_gym.make(self.env_id)
        self.env = mo_gym.LinearReward(self.env)

        self.obs_dim,self.act_dim = self.env.observation_space.shape[0],self.env.action_space.shape[0]
        self.reward_factor = configs["reward_factor"]  
        self.style_dim = len(self.reward_factor.keys())
        self.seed = seed
        self.agent = agent_type()
        #! PARA
        self.gamma = configs["gamma"]
        self.tau = configs["tau"]
        self.max_traj_len = configs["max_traj_len"]

        #! ENV STATE
        self.done = True 
        self.state = None
        self.style_state = None 
        self.reward_weight = None

        
        #! Statistics
        self.episode_reward = 0
        self.episode_reward_vec = np.zeros(shape=(self.style_dim,))
        self.episode_length = 0

        self.episode_reward_list = []
        self.episode_reward_vec_list = []
        self.episode_length_list = []
        self.episode_task_id = []


        self.gae = GAE()

        #! task to rollout
        self.weight_list = [np.array([1.0 for _ in range(self.style_dim)])]
        self.style_list = [np.array([1.0 for _ in range(self.style_dim)])]
        self.task_id_list = [0]
    
    def get_tasks(self,task_id,weight_list,style_list):
        self.task_id_list = task_id
        self.weight_list = weight_list
        self.style_list = style_list

    def fetch_model(self,path,name):
        self.agent.fetch_model_parameter(path,name)
    
    def step(self,action):
        next_state,scalar_reward,done,_,info = self.env.step(action)
        reward_vec = info["vector_reward"]
        self.episode_reward += scalar_reward 
        self.episode_reward_vec += reward_vec 
        self.episode_length += 1
        self.done = done 
        self.state = next_state
        return next_state,scalar_reward,reward_vec,done,info

    def reset(self):
        #! 重新设定当前 reward weight
        #! 重新设定当前 preference 
        _id = np.random.choice(range(len(self.weight_list)))
        reward_weight = self.weight_list[_id]
        style_state = self.style_list[_id]  
        self.task_id = self.task_id_list[_id]
        #reward_weight,reward_factor,style_state = generate_reward_factors(self.reward_factor,self.seed)
        # reward_weight: 权重向量
        # reward_factor: 权重字典
        # style_state: 风格状态
        self.env.set_weight(reward_weight)
        state,info = self.env.reset()
        self.done = False 
        self.reward_weight = reward_weight  

        return state,style_state,reward_weight
    
    def get_input(self):
        state = self.agent.process_state(self.state)
        style_state = self.agent.process_style(self.style_state)
        return state,style_state

    def sample_one_traj(self):
        memory = []
        num_steps = 0
        while num_steps < self.max_traj_len:
            if self.done:
                self._flush_statistics()
                self.state,self.style_state,self.reward_weight = self.reset()
            else:
                state,style_state = self.get_input()
                mean,logstd,probs,value = self.agent.get_model_result(state,style_state)
                action = self.agent.get_action(probs)
                old_logprob = probs.log_prob(action).sum(-1)
                action = action.squeeze(0).numpy()
                next_state,scalar_reward,reward_vec,done,info = self.step(action)

                num_steps += 1 

                instance = Instance(
                    state = state.squeeze(0).numpy(),
                    style = style_state.squeeze(0).numpy(),
                    reward_weight=self.reward_weight,
                    action = action,
                    old_log_prob = old_logprob.squeeze(0).numpy(),
                    old_state_value = value.squeeze(0).numpy(),
                    is_done = done,
                    reward = scalar_reward,
                    reward_vec = reward_vec,

                )
                # print("Check: ", instance.state.shape,instance.style.shape,
                #       instance.action.shape,instance.old_log_prob.shape,instance.old_state_value.shape,
                #       instance.reward_vec.shape,instance.reward_weight.shape,instance.reward.shape)
                memory.append(instance)
        
        state,style_state = self.get_input()
        _,_,_,value = self.agent.get_model_result(state,style_state) 
        bootstrap_value = value.squeeze(0).numpy() if not self.done else np.zeros_like(value.squeeze(0).numpy())
        rewards_vec_list = [j.reward_vec for j in memory]
        mask_list = [j.is_done for j in memory]
        value_list = [j.old_state_value for j in memory]
        advantages, returns = self.gae.estimate_advantages_vec(
                    rewards_vec_list, mask_list, value_list, self.gamma, self.tau, bootstrap_value)
        
        for index, item in enumerate(memory):
            item.advantage = advantages[index]
            item.q_value = returns[index]
        
        return memory, self.get_statistics()
    def _flush_statistics(self):
        self.episode_length_list.append(self.episode_length)
        self.episode_reward_list.append(self.episode_reward)
        self.episode_reward_vec_list.append(self.episode_reward_vec)
        self.episode_task_id.append(self.task_id)

        self.episode_reward = 0
        self.episode_reward_vec = np.zeros(shape=(self.style_dim,))
        self.episode_length = 0
    
    def get_statistics(self):  
        ep_length = self.episode_length_list
        ep_rew = self.episode_reward_list
        ep_rew_vec = self.episode_reward_vec_list 
        self.episode_length_list = []
        self.episode_reward_list = []
        self.episode_reward_vec_list = []
        res = {
            "Episode/length":ep_length,
            "Episode/reward":ep_rew,
            "Episode/task_id":self.episode_task_id,
            # "episode_reward_vec":ep_rew_vec
        }
        for i in range(self.style_dim):
            res[f"Episode/return_{i}"] = [tmp[i] for tmp in ep_rew_vec]
        return res

@ray.remote
class RemoteWorker(Worker):
    def __init__(self, configs, agent_type: PPO_Agent, seed: int) -> None:
        super().__init__(configs, agent_type, seed)


    

    
    