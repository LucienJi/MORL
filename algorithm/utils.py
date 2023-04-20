import mo_gymnasium as mo_gym
import gym 
import numpy as np
import random
import dataclasses

def make_styled_env(env_id,seed,reward_factors):
    assert type(reward_factors) == list 
    def thunk():
        env = mo_gym.make(env_id)
        env = mo_gym.LinearReward(env, weight=np.array(reward_factors))
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = mo_gym.MORecordEpisodeStatistics(env)
        if hasattr(env, 'seed'):
            env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def generate_reward_factors(factors:dict,seed = 0):
    random.seed(seed)
    reward_factors = {}
    reward_factors_list = []
    style_state = []
    for factor_name, factor_info in factors.items():
        if factor_info['fine_grit'] > 0:
            if factor_info['lower_bound'] == factor_info['upper_bound']:
                reward_factors[factor_name] = factor_info['lower_bound']
                reward_factors_list.append(factor_info['lower_bound'])
            else:
                grit_factor = np.linspace(
                    factor_info['lower_bound'],
                    factor_info['upper_bound'],
                    int((factor_info['upper_bound'] - factor_info['lower_bound']) / factor_info['fine_grit']) + 1
                    )
                reward_factors[factor_name] = random.choice(grit_factor)
                reward_factors_list.append(reward_factors[factor_name])
        else:
            reward_factors[factor_name] = \
                random.uniform(factor_info['lower_bound'], factor_info['upper_bound'])
            reward_factors_list.append(reward_factors[factor_name])

        if factor_info['lower_bound'] == factor_info['upper_bound']:
            style_value= factor_info['upper_bound'] / 2
        else:
            style_value = (reward_factors[factor_name] - factor_info['lower_bound']) / \
                                (factor_info['upper_bound'] - factor_info['lower_bound'])
                
        style_state.append(style_value)
    return reward_factors_list, reward_factors,style_state


class MO_Stats(object):
    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self.actions = []
        self.states = []
        self.reward_vectors = []
    
    def update(self,s,a,r):
        self.actions.append(a)
        self.states.append(s)
        self.reward_vectors.append(r)
    
    def _analyze(self,data):
        data = np.stack(data,axis=0) #(N,dim)
        if len(data.shape) == 1:
            data = data.reshape(-1,1)
        n_dim = data.shape[1]
        if data.dtype == np.int64:
            min_value,max_value = np.min(data),np.max(data)
            bin_num = max_value - min_value + 1
        else:
            bin_num = 10
        result = []
        for i in range(n_dim):
            data_i = data[:,i]
            hist,bin_edges = np.histogram(data_i,bins=bin_num,density=True)
            result.append(hist)
        return np.stack(result,axis=0)
    def analyze(self):
        """
        return: distribution of actions,states,reward_vectors
        shape (n_dim,n_bin)
        """
        act_dist = self._analyze(self.actions)
        states_dist = self._analyze(self.states)
        rew_dist = self._analyze(self.reward_vectors)
        return act_dist,states_dist,rew_dist

    def raw_data(self):
        actions = np.stack(self.actions,axis=0)
        if len(actions.shape) == 1:
            actions = actions.reshape(-1,1)
        states = np.stack(self.states,axis=0)
        reward_vectors = np.stack(self.reward_vectors,axis=0)
        return actions,states,reward_vectors
        

