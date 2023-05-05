import mo_gymnasium as mo_gym
import gym 
import numpy as np
import random
import dataclasses
import os 

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

def non_dominated_sort(rewards):
    """
    input: rewards (N,dim)
    output: rank (N,)
    """
    N = len(rewards)
    rank = np.zeros(N,dtype=np.int32)
    rank_dict = dict()
    S = [[] for _ in range(N)]
    n = np.zeros(N,dtype=np.int32)
    rank_dict[1] = []
    for p in range(N):
        for q in range(N):
            if np.all(rewards[p] >= rewards[q]) and np.any(rewards[p] > rewards[q]):
                ## p dominates q 
                if q not in S[p]:
                    S[p].append(q)
            elif np.all(rewards[p] <= rewards[q]) and np.any(rewards[p] < rewards[q]):
                ## q dominates p, to note 可以存在两个element 不是相互 dominate 的情况
                n[p] += 1
        if n[p] == 0:
            rank[p] = 1
            rank_dict[1].append(p)
            if p not in S[p]: #! 没觉得应该加这句 
                S[p].append(p)
    i = 1
    while np.any(rank == i):
        Q = []
        for p in range(N):
            if rank[p] == i:
                for q in S[p]:
                    n[q] -= 1 #! 这里假如 S[p] 中的元素是 p 自身，那么 n[q] 会被减两次，这里应该加一个判断
                    if n[q] == 0: #! 这里 n[p] = -1  了
                        rank[q] = i + 1
                        if q not in Q:
                            Q.append(q)
        i += 1
        rank_dict[i] = Q
    return rank,rank_dict
class Factor_Sampler(object):
    def __init__(self,factors:dict) -> None:
        self.factors = factors
        self.n_style = len(factors.keys())
        self.style_nums = []
        self.style_list = []
        self.style_states = []
        for factor_name,factor_info in factors.items():
            tmp_list = []
            tmp_style_states = []
            assert factor_info['fine_grit'] > 0
            if factor_info['lower_bound'] == factor_info['upper_bound']:
                tmp_list.append(np.round(factor_info['lower_bound'],4))
                tmp_style_states.append(factor_info['upper_bound'] / 2)
            else:
                grit_factor = np.linspace(
                    factor_info['lower_bound'],
                    factor_info['upper_bound'],
                    int((factor_info['upper_bound'] - factor_info['lower_bound']) / factor_info['fine_grit']) + 1
                    )
                grit_factor = np.round(grit_factor,4)
                tmp_list = grit_factor.tolist()
                tmp_style_states = (grit_factor - factor_info['lower_bound'])/ \
                    (factor_info['upper_bound'] - factor_info['lower_bound'])
            self.style_nums.append(len(tmp_list))
            self.style_list.append(tmp_list)
            self.style_states.append(tmp_style_states)
        
        self.total_combination = np.prod(self.style_nums)
        
        self._num_helpers = [1]
        for i in range(1,self.n_style):
            self._num_helpers.append(np.prod(self.style_nums[:i]))

        #! 目前尝试记录 update 前后的多次 return 均值，从而计算 weights
        #! 保留 update 前的均值；保留上次 update 后每个 weight 被更新的次数
        self.old_returns = np.zeros((self.total_combination,self.n_style))
        self.current_returns = np.zeros((self.total_combination,self.n_style))
        self.update_times = np.zeros((self.total_combination,))
        self.style_weights = np.zeros((self.total_combination,self.n_style))

        ## ? debugs: 
        self._debugs = np.zeros((self.total_combination,))

    def _pair_to_weight(self,idx:list):
        assert len(idx) == self.n_style
        return [self.style_list[i][idx[i]] for i in range(self.n_style)]
    
    def _pair_to_styles(self,idx:list):
        assert len(idx) == self.n_style
        return [self.style_states[i][idx[i]] for i in range(self.n_style)]
    
    def _weight_to_pair(self,weights):
        assert len(weights) == self.n_style
        result = []
        for i in range(self.n_style):
            result.append(self.style_list[i].index(weights[i]))
        return result
    
    def _pair_to_num(self,idx:list):
        assert len(idx) == self.n_style
        return sum([self._num_helpers[i]*idx[i] for i in range(self.n_style)])
    
    def _num_to_pair(self,num:int):
        assert num < self.total_combination
        result = []
        for i in reversed(range(self.n_style)):
            result.append(num // self._num_helpers[i])
            num = num % self._num_helpers[i]
        result.reverse()
        return result
    def _calc_weights(self,clip = True,norm = True):
        weights = self.style_weights
        if norm:
            normalized_weights = (weights - weights.mean(0,keepdims=True)) / weights.std(0,keepdims=True)
            normalized_weights = normalized_weights.sum(-1)
            weights = normalized_weights
        else:
            weights = weights.sum(-1)
        if clip:
            return np.clip(weights,a_min=0,a_max=None)
        else:
            return weights
    
    def update_returns(self,weight,returns):
        idx = self._weight_to_pair(weight)
        num = self._pair_to_num(idx)
        self.current_returns[num] += returns
        self.update_times[num] += 1
    def update_weights(self,tau = 0.9):
        bonus = 1.0
        self.update_times += 1e-6
        self.current_returns = self.current_returns / self.update_times.reshape(-1,1)
        delta_returns = self.current_returns - self.old_returns 
        self.old_returns = self.current_returns * (1 - tau) + self.old_returns * tau
        #! 可以这里直接修改 weights，也可以同时检查没有 update 的weights，加上额外的权重
        self.style_weights = np.zeros((self.total_combination,self.n_style))
        self.style_weights += delta_returns
        self.style_weights[np.where(self.update_times<1)] += 1.0  * bonus
        #! reset return track
        self.current_returns = np.zeros((self.total_combination,self.n_style))
        self.update_times = np.zeros((self.total_combination,))


    # def update_weights(self,weight,delta_ret):
    #     idx = self._weight_to_pair(weight)
    #     num = self._pair_to_num(idx)
    #     if type(delta_ret) == list:
    #         delta_ret = np.array(delta_ret)
    #     assert delta_ret.shape == (self.n_style,)
    #     self.style_weights[num] = delta_ret
    
    def sample_weights(self,num_samples,clip=False,norm=False):
        weights = self._calc_weights(clip=clip,norm=norm)
        weights = np.exp(weights) / np.exp(weights).sum()
        sampled = np.random.choice(self.total_combination,size=num_samples,p=weights)
        sampled = [self._pair_to_weight(self._num_to_pair(i)) for i in sampled]
        return sampled
    def generate_reward_factors(self,num_samples,clip=False,norm=False):
        weights = self._calc_weights(clip=clip,norm=norm)
        weights = np.exp(weights) / np.exp(weights).sum()
        sampled_idx = np.random.choice(self.total_combination,size=num_samples,p=weights)
        ## ? debugs:
        self._debugs[sampled_idx] += 1
        sampled_pair = [self._num_to_pair(i) for i in sampled_idx]
        sampled_weights = [self._pair_to_weight(i) for i in sampled_pair]
        sampled_styles = [self._pair_to_styles(i) for i in sampled_pair]
        return sampled_weights,sampled_styles
    
    def _save_debugs(self,path):
        d = dict(
            sample_times = self._debugs,
            style_weights = self.style_weights,
            old_returns = self.old_returns,
        )
        data_path = os.path.join(path,"sampler_debug")
        np.savez(data_path,**d)


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
        

