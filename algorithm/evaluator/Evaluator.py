import mo_gymnasium as mo_gym
import numpy as np 
from itertools import product
import os 
"""
先做一个离散版的，通常训练过程用离散版的应该没啥问题
"""

def weight_to_style(weight):
    """这里不使用 preference，和"""

class DiscreteFactorSampler(object):
    """由于细分 reward weight 会指数型上涨，所以不维护每一个 reward weight 的状态"""
    def __init__(self,factors) -> None:
        self.factors = factors
        self.n_style = len(factors.keys())
        
        self.factor_names = []
        self.reward_weights = [] #! reward_name: 可以选择的值
        self.reward_styles = []
        self.lower_bounds = []
        self.upper_bounds = []
        self.pair2ID = {}
        self.ID2pair = {}

        for factor_name,factor_info in factors.items():
            tmp_list = []
            tmp_style_states = []
            assert factor_info['fine_grit'] > 0
            if factor_info['lower_bound'] == factor_info['upper_bound']: #! 这个 reward 不做区分
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

                #! weight 转 style state 
                tmp_style_states = (grit_factor - factor_info['lower_bound'])/ \
                    (factor_info['upper_bound'] - factor_info['lower_bound'])
            self.reward_weights.append(tmp_list)
            self.reward_styles.append(tmp_style_states)
            self.factor_names.append(factor_name)
            self.lower_bounds.append(factor_info['lower_bound'])
            self.upper_bounds.append(factor_info['upper_bound'])
        
        self.calc_n_weight()
    
    def sample_weight(self,n):
        """这里的 n 是指需要采样的 reward weight 的数量"""
        ids = np.random.choice(self.n_weight,n)
        weights = []
        styles = []
        task_ids = []
        for id in ids:
            w = self._ID2weight(id)
            weights.append(w)
            styles.append(self.weight2style(w))
            task_ids.append(id)
        return task_ids,weights,styles

    def generate_reward_weight(self):
        pair = []
        weight = []
        for i in range(self.n_style):
            w = np.random.choice(self.reward_weights[i])
            weight.append(w)
            pair.append(self.reward_weights[i].index(w))
        weight = np.array(weight)
        id = self.pair2ID[tuple(pair)]
        style = self.weight2style(weight)
        return id,weight,style

    def _weight2ID(self,weight):
        pair = []
        for i in range(self.n_style):
            pair.append(self.reward_weights[i].index(weight[i]))
        return self.pair2ID[tuple(pair)]
    def _ID2weight(self,ID):
        ## 只在 class 内部使用，外部只需要 使用 task id 
        pair = self.ID2pair[ID]
        weight = []
        for i in range(self.n_style):
            weight.append(self.reward_weights[i][pair[i]])
        return np.array(weight)
    
    def calc_n_weight(self):
        nums = [len(i) for i in self.reward_weights]
        self.n_weight = np.prod(nums)
        
        ids = product(*[range(i) for i in nums])
        for i,id in enumerate(ids):
            self.pair2ID[id] = i 
            self.ID2pair[i] = id
    
    #! 这里之后需要重新构造一个，用于生成 style state 的函数
    def weight2style(self,weight):
        style = []
        for i in range(self.n_style):
            style.append((weight[i] - self.lower_bounds[i]) / (self.upper_bounds[i] - self.lower_bounds[i]))
        return np.array(style).reshape(-1) 

    @property
    def n_tasks(self):
        return self.n_weight

class Evaluator(object):
    def __init__(self,configs,logger = None) -> None:
        self.configs = configs
        self.env_id = configs['env_id']    
        self.env = mo_gym.make(self.env_id,configs['max_episode_steps'])
        self.env = mo_gym.LinearReward(self.env)

        self.obs_dim,self.act_dim = self.env.observation_space.shape[0],self.env.action_space.shape[0]
        self.reward_factor = configs["reward_factor"]  
        self.style_dim = len(self.reward_factor.keys())
        self.model_path = configs["model_path"]
        self.logger = logger

        self.reward_sampler = DiscreteFactorSampler(self.reward_factor)
    
    def sample_tasks(self,n):
        """
        返回 task id, reward weight, style state
        """
        return self.reward_sampler.sample_weight(n)

    def evaluate_all_tasks(self):
        """
        评估所有的 task
        """
        n_task = self.reward_sampler.n_tasks
        all_task_ids = list(range(n_task))
        all_weights = []
        all_styles = []
        for task_id in all_task_ids:
            weight = self.reward_sampler._ID2weight(task_id)
            style = self.reward_sampler.weight2style(weight)
            all_weights.append(weight)
            all_styles.append(style)
        return all_task_ids,all_weights,all_styles

    
    def calc_statistics(self,list_statistics:list):
        """目前是从 rollout 中收集的 statistics"""
        res= {}
        for k,v in list_statistics[0].items():
            res[k] = []
        for data in list_statistics:
            for k,v in data.items():
                res[k].extend(v)
        
        np.savez_compressed(os.path.join(self.model_path,"statistics.npz"),**res)
        return res 

        