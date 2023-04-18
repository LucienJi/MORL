import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy 
from torch.distributions.categorical import Categorical
import os 
def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class Block(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_size = [256,256]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        fc_linear = []
        fc_linear.append(nn.Linear(input_dim,hidden_size[0]))
        fc_linear.append(nn.Tanh())
        for i in range(len(hidden_size) - 1):
            fc_linear.append(nn.Linear(hidden_size[i],hidden_size[i+1]))
            fc_linear.append(nn.Tanh())
        fc_linear.append(nn.Linear(hidden_size[-1],output_dim))
        self.net = nn.Sequential(*fc_linear)
    
    def forward(self,x):
        return self.net(x) 

class ControlNet(nn.Module):
    def __init__(self,parent_block:nn.Module,extra_input_dim,allow_retrain = False):
        super().__init__()
        self.locked_block = parent_block
        with torch.no_grad():
            for p in self.locked_block.parameters():
                p.requires_grad = allow_retrain 
        assert hasattr(self.locked_block,"input_dim") and hasattr(self.locked_block,"output_dim")
        self.trainable_block = copy.deepcopy(self.locked_block)
        with torch.no_grad():
            for p in self.trainable_block.parameters():
                p.requires_grad = True
        input_dim = self.locked_block.input_dim + extra_input_dim

        self.zero_layer1 = nn.Linear(input_dim,self.locked_block.input_dim)
        self.zero_layer2 = nn.Linear(self.locked_block.output_dim,self.locked_block.output_dim)
    
    def init(self):
        for m in self.modules():
            if m in (self.zero_layer1,self.zero_layer2):
                constant_init(m,val=0.0)
    def forward(self,x,extra_input):
        raw_output = self.locked_block.forward(x)
        augmented_x = torch.cat((x,extra_input),dim=-1)
        x = x + self.zero_layer1(augmented_x)
        output = self.zero_layer2(self.trainable_block(x))
        return raw_output + output 

    def load_expert_state_dict(self,state_dict):
        self.locked_block.load_state_dict(state_dict)
        self.trainable_block.load_state_dict(state_dict)
        with torch.no_grad():
            for p in self.trainable_block.parameters():
                p.requires_grad = True
        with torch.no_grad():
            for p in self.locked_block.parameters():
                p.requires_grad = False 


class Expert(object):
    def __init__(self,obs_dim,act_dim) -> None:
        obs_dim,act_dim = obs_dim,act_dim
        self.critic = Block(obs_dim,1,hidden_size=[256,256])
        self.actor = Block(obs_dim,act_dim,hidden_size=[256,256])
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    def save(self,path,name = "expert"):
        if not os.path.exists(path):
            os.makedirs(path)
        model_path = os.path.join(path,name)
        torch.save(self.critic.state_dict(),model_path + "_critic.pt")
        torch.save(self.actor.state_dict(),model_path + "_actor.pt")
    def load(self,path,name = 'expert'):
        assert os.path.exists(path)
        model_path = os.path.join(path,name)
        self.critic.load_state_dict(torch.load(model_path + "_critic.pt"))
        self.actor.load_state_dict(torch.load(model_path + "_actor.pt"))
    def to(self,device):
        self.critic.to(device)
        self.actor.to(device)
        return self


class StyleExpert(object):
    def __init__(self,obs_dim,act_dim,style_dim,parent:Expert = None,allow_retrain = False) -> None:
        obs_dim,act_dim = obs_dim,act_dim
        if parent is not None:
            self.style_critic = ControlNet(parent.critic,extra_input_dim=style_dim,allow_retrain=allow_retrain)
            self.style_actor = ControlNet(parent.actor,extra_input_dim=style_dim,allow_retrain=allow_retrain)
        else:
            self.style_critic = ControlNet(Block(obs_dim,1,hidden_size=[256,256]),extra_input_dim=style_dim,allow_retrain=allow_retrain)
            self.style_actor = ControlNet(Block(obs_dim,act_dim,hidden_size=[256,256]),extra_input_dim=style_dim,allow_retrain=allow_retrain)
        self.style_actor.init()
        self.style_critic.init()
    def to(self,device):
        self.style_actor.to(device)
        self.style_critic.to(device)
        return self
    def get_value(self, x,style):
        return self.style_critic(x,style)

    def get_action_and_value(self, x,style, action=None):
        logits = self.style_actor(x,style)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.style_critic(x,style)
    def load_expert(self,path,name):
        if os.path.exists(path):
            model_path = os.path.join(path,name)
            critic_path = model_path + '_critic.pt'
            actor_path = model_path + '_actor.pt'
            self.style_critic.load_expert_state_dict(torch.load(critic_path))
            self.style_actor.load_expert_state_dict(torch.load(actor_path))
            print(f"Model Loaded From: {model_path}") 
        else:
            print("Path Not Exist")    

    def save(self,path,name):
        if not os.path.exists(path):
            os.makedirs(path)
        model_path = os.path.join(path,name)        
        torch.save(self.style_critic.state_dict(),model_path + "_critic.pt")
        torch.save(self.style_actor.state_dict(),model_path + "_actor.pt")
    def load(self,path,name):
        assert os.path.exists(path)
        model_path = os.path.join(path,name)
        self.style_critic.load_state_dict(torch.load(model_path + "_critic.pt"))
        self.style_actor.load_state_dict(torch.load(model_path + "_actor.pt"))








