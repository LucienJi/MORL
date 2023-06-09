import numpy as np
import copy
import time
import torch as th
import mo_gymnasium as mo_gym

class Instance:
    def __init__(
            self,
            data_time=None,
            state=None,
            style = None,
            reward_weight=None,
            action=None,
            old_log_prob=None,
            old_state_value=None,
            advantage=None,
            q_value = None,
            is_done=None,
            reward=None,
            reward_vec = None 
            ):

        if data_time is None:
            self.data_time = time.time()
        else:
            self.data_time = data_time
        self.state = state
        self.style = style 
        self.reward_weight = reward_weight
        self.action = action
        self.old_log_prob = old_log_prob
        self.old_state_value = old_state_value
        self.advantage = advantage
        self.is_done = is_done
        self.reward = reward
        self.reward_vec = reward_vec
        self.q_value = q_value  


class TrainingSet:
    def __init__(
            self,
            configs,
    ):  
        self.env_id = configs['env_id']    
        self.env = mo_gym.make(self.env_id,configs['max_episode_steps'])
        self.env = mo_gym.LinearReward(self.env)

        state_dim,act_dim = self.env.observation_space.shape[0],self.env.action_space.shape[0]
        
        max_capacity = configs["max_capacity"] 
        self.reward_factor = configs["reward_factor"]
        style_dim = len(self.reward_factor.keys())


        self.data_time_list = np.zeros(shape=(max_capacity,))
        self.state_list = np.zeros(shape=(max_capacity,state_dim))
        self.style_list = np.zeros(shape=(max_capacity,style_dim))
        self.reward_weight_list = np.zeros(shape=(max_capacity,style_dim))
        self.action_list = np.zeros(shape=(max_capacity,act_dim))
        self.old_log_prob_list = np.zeros(shape=(max_capacity,))
        self.old_state_value_list = np.zeros(shape=(max_capacity,style_dim))
        self.advantage_list = np.zeros(shape=(max_capacity,style_dim))
        self.q_value_list = np.zeros(shape=(max_capacity,style_dim))
        self.ptr = 0
        self._size = 0

        self.max_capacity = max_capacity
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.style_dim = style_dim

    def clear(self):
        max_capacity = self.max_capacity
        state_dim = self.state_dim
        act_dim = self.act_dim
        style_dim = self.style_dim
        self.data_time_list = np.zeros(shape=(max_capacity,))
        self.state_list = np.zeros(shape=(max_capacity,state_dim))
        self.style_list = np.zeros(shape=(max_capacity,style_dim))
        self.reward_weight_list = np.zeros(shape=(max_capacity,style_dim))
        self.action_list = np.zeros(shape=(max_capacity,act_dim))
        self.old_log_prob_list = np.zeros(shape=(max_capacity,))
        self.old_state_value_list = np.zeros(shape=(max_capacity,style_dim))
        self.advantage_list = np.zeros(shape=(max_capacity,style_dim))
        self.q_value_list = np.zeros(shape=(max_capacity,style_dim))
        self.ptr = 0
        self._size = 0
    def len(self):
        return self._size

    def append_instance(self, instances):
        for instance in instances:
            self.data_time_list[self.ptr] = instance.data_time
            self.state_list[self.ptr] = instance.state
            self.style_list[self.ptr] = instance.style
            self.reward_weight_list[self.ptr] = instance.reward_weight
            
            self.action_list[self.ptr] = instance.action
            self.old_log_prob_list [self.ptr] = instance.old_log_prob

            self.old_state_value_list[self.ptr] = instance.old_state_value
            self.advantage_list[self.ptr] = instance.advantage
            self.q_value_list[self.ptr] = instance.q_value
            self.ptr = (self.ptr + 1) % self.max_capacity
            self._size = min(self._size + 1, self.max_capacity)

    def slice(self, index_list,batch_size = None):
        if index_list is None:
            index_list = self._generate_random_index(batch_size)
        slice_dict = {}

        batch_size = len(index_list)
        slice_dict["states"] = self.state_list[index_list].reshape(batch_size, -1)
        slice_dict["styles"] = self.style_list[index_list].reshape(batch_size, -1)
        slice_dict["reward_weights"] = self.reward_weight_list[index_list].reshape(batch_size, -1)
        slice_dict["actions"] = self.action_list[index_list].reshape(batch_size, -1)
        
        slice_dict["old_log_prob"] = self.old_log_prob_list[index_list].reshape(batch_size,)
        
        slice_dict["old_state_value"] = self.old_state_value_list[index_list].reshape(batch_size, -1)
        slice_dict["advantages"] = self.advantage_list[index_list].reshape(batch_size, -1)
        slice_dict["q_values"] = self.q_value_list[index_list].reshape(batch_size, -1)
        return slice_dict


    def _generate_random_index(self, batch_size):
        return np.random.choice(range(self.len()), batch_size, replace=False)
    
    def get_training_index(self,minibatch_size):
        indices = np.random.permutation(self.len())
        start = 0
        end = min(minibatch_size,self.len())

        while end <= self.len():
            yield indices[start:end]
            if end == self.len():break
            start = end 
            end = min(end + minibatch_size,self.len())