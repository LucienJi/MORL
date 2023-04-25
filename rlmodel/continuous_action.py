import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
from torch.distributions.normal import Normal
from .controlnet import StyleExpert,Expert

class ContinuousStyleExpert(StyleExpert):
    def __init__(self, obs_dim, act_dim, style_dim, allow_retrain,hidden_size = [512,512]) -> None:
        super().__init__(obs_dim, act_dim * 2, style_dim, allow_retrain,hidden_size =hidden_size)
        #! 这里我们 act dim * 2, 目的是为了让 actor 输出的是 mean 和 std
    
    def get_action_and_value(self, x, style, action=None):
        mean_logstd = self.actor(x, style)
        mean,logstd = mean_logstd.chunk(2, dim=-1)
        std = logstd.exp()
        probs = Normal(mean, std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), self.critic(x, style)

class ContinuousExpert(Expert):
    def __init__(self, obs_dim, act_dim,hidden_size = [512,512]) -> None:
        super().__init__(obs_dim, act_dim * 2,hidden_size = hidden_size)
        #! 这里我们 act dim * 2, 目的是为了让 actor 输出的是 mean 和 std
    
    def get_action_and_value(self, x,style, action=None):
        mean_logstd = self.actor(x)
        mean,logstd = mean_logstd.chunk(2, dim=-1)
        std = logstd.exp()
        probs = Normal(mean, std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), self.critic(x)