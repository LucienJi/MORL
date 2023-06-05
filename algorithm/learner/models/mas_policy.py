import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy 
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import os 
LOG_STD_MAX = 2
LOG_STD_MIN = -20
def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class PolicyNet(nn.Module):
    def __init__(self, state_dim, act_dim,style_dim):
        super(PolicyNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim + style_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU()
        )
        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, style_dim)
        )
        self.policy_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
        )
        self.mean_head = nn.Linear(256, act_dim)
        self.logstd_head = nn.Linear(256, act_dim)

    def init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal(m.weight)

    def forward(self, state,style):
        # ma: action mask: 1 means invalid!!
        state = torch.cat([state,style],dim = 1)
        backbone = self.mlp(state)
        policy_out = self.policy_head(backbone)
        mean,logstd = self.mean_head(policy_out),self.logstd_head(policy_out)
        # probs = Normal(mean, logstd.exp())
        # action = probs.sample()
        # return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), self.value_head(backbone)
        return mean,logstd,self.value_head(backbone)

