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
    def __init__(self,env_id,
                 agent_type:PPO_Agent,
                 reward_factor:dict,
                 seed:int) -> None:
        self.env_id = env_id    
        self.env = mo_gym.make(env_id)
        self.env = mo_gym.LinearReward(self.env)

        self.obs_dim,self.act_dim = self.env.observation_space.shape[0],self.env.action_space.shape[0]
        self.reward_factor = reward_factor  
        self.style_dim = len(self.reward_factor.keys())
        self.seed = seed
        self.agent = agent_type()
        #! PARA
        self.gamma = 0.99
        self.tau = 0.95

        #! ENV STATE
        self.done = True 
        self.state = None
        self.style_state = None 
        self.reward_weight = None

        self.max_traj_len = 128
        #! Statistics
        self.episode_reward = 0
        self.episode_reward_vec = np.zeros(shape=(self.style_dim,))
        self.episode_length = 0
        self.gae = GAE()

    def fetch_model(self,path):
        self.agent.fetch_model_parameter(path)
    
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
        reward_weight,reward_factor,style_state = generate_reward_factors(self.reward_factor,self.seed)
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
                self._reset_statistics()
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
        
        return memory 
    def _reset_statistics(self):
        self.episode_reward = 0
        self.episode_reward_vec = np.zeros(shape=(self.style_dim,))
        self.episode_length = 0

@ray.remote
class RemoteWorker(Worker):
    def __init__(self, env_id, agent_type: PPO_Agent, reward_factor: dict, seed: int) -> None:
        super().__init__(env_id, agent_type, reward_factor, seed)


    

    
    