import torch as th 
import numpy as np 
from algorithm.learner.model_utils import load_script_model
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
LOG_STD_MAX = 2
LOG_STD_MIN = -20
class PPO_Agent(object):
    def __init__(self) -> None:
        self.sampler_method = 'prob'
        self.pnet = None 
    
    def process_state(self,state):
        return th.from_numpy(state).float().unsqueeze(0)
    def process_style(self,style):
        return th.from_numpy(style).float().unsqueeze(0)
    def fetch_model_parameter(self,path):
        self.pnet = load_script_model(path)
    
    def get_model_result(self,state,style):
        # print("Check: ", state.shape,style.shape)
        with th.no_grad():
            mean,logstd,value = self.pnet(state,style)
            logstd = th.clamp(logstd, LOG_STD_MIN, LOG_STD_MAX)
        probs = Normal(mean,logstd.exp())
        return mean,logstd,probs,value
    
    def get_action(self,probs):
        if self.sampler_method == 'prob':
            action = probs.sample()
        elif self.sampler_method == 'greedy':
            action = probs.mean
        else:
            raise NotImplementedError
        return action