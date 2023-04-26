import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy 
from torch.distributions.categorical import Categorical
import os 

from .controlnet import Block,ControlBlock,ControlNet,MLPNet

#! Base Case，Single Style Model

class VMLPNet(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_size = [256,256],extra_input_dim = 0):
        super(VMLPNet,self).__init__()
        self.net = Block(input_dim = input_dim + extra_input_dim,
                         output_dim = hidden_size[-1],hidden_size=hidden_size
                         )
        #! 这里是简单的 multi-head 设定，可以再复杂化
        self.output_dim = output_dim
        self.output_layer = nn.Linear(hidden_size[-1],output_dim)
    
    def forward(self,x,extra_input = None,weights = None):
        if extra_input is not None:
            x = torch.cat([x,extra_input],dim = 1)
        x = self.net.forward(x)
        x = self.output_layer(x)
        if weights is not None:
            ## x.shape (bz,output_dim)
            ## weights (output_dim)
            if weights.ndim == 1:
                weights = weights.unsqueeze(0)
            x = torch.sum(x * weights,dim = 1,keepdim = True)
        return x
    def separete_save(self,path,name):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.net.state_dict(),path + f"/{name}_block.pt")
        torch.save(self.output_layer.state_dict(),path + f"/{name}_outlayer.pt")

    def load(self,path,name):
        self.net.load_state_dict(torch.load(path + f"/{name}_block.pt"))
        self.output_layer.load_state_dict(torch.load(path + f"/{name}_outlayer.pt"))

class VMLPNet_v2(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_size = [256,256],extra_input_dim = 0):
        super(VMLPNet,self).__init__()
        self.net = Block(input_dim = input_dim + extra_input_dim,
                         output_dim = hidden_size[-1],hidden_size=hidden_size
                         )
        #! 这里是简单的 multi-head 设定，可以再复杂化
        self.output_dim = output_dim

        self.output_layers = nn.ModuleList()
        for i in range(output_dim):
            self.output_layers.append(nn.Sequential(
                nn.Linear(hidden_size[-1],hidden_size[-1]),
                nn.Tanh(),
                nn.Linear(hidden_size[-1],1)
            ))
        
    def _multi_head(self,x):
        output = []
        for i in range(self.output_dim):
            output.append(self.output_layers[i](x))
        output = torch.cat(output,dim = -1)
        return output
    
    def forward(self,x,extra_input = None,weights = None):
        if extra_input is not None:
            x = torch.cat([x,extra_input],dim = 1)
        x = self.net.forward(x)
        x = self._multi_head(x)
        if weights is not None:
            ## x.shape (bz,output_dim)
            ## weights (output_dim)
            if weights.ndim == 1:
                weights = weights.unsqueeze(0)
            x = torch.sum(x * weights,dim = 1,keepdim = True)
        return x
    def separete_save(self,path,name):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.net.state_dict(),path + f"/{name}_block.pt")
        torch.save(self.output_layers.state_dict(),path + f"/{name}_outlayer.pt")

    def load(self,path,name):
        self.net.load_state_dict(torch.load(path + f"/{name}_block.pt"))
        self.output_layers.load_state_dict(torch.load(path + f"/{name}_outlayer.pt"))

class VControlNet(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_size = [256,256],use_mas = False,extra_input_dim = 0,allow_retrain = False):
        super().__init__()
        self.use_mas = use_mas
        self.extra_intput_dim = extra_input_dim
        if self.use_mas:
            self.net = ControlBlock(input_dim,hidden_size[-1],extra_input_dim = extra_input_dim,hidden_size=hidden_size,allow_retrain = allow_retrain)
        else:
            self.net = Block(input_dim,hidden_size[-1],hidden_size = hidden_size)
        self.output_layer = nn.Linear(hidden_size[-1],output_dim) # 这一部分也是 trainable 的
    def forward(self,x,extra_input = None,weights = None):
        if self.use_mas:
            x = self.net.forward(x,extra_input)
        else:
            x = self.net.forward(x)
        x = self.output_layer(x)
        if weights is not None:
            ## x.shape (bz,output_dim)
            ## weights (output_dim)
            if weights.ndim == 1:
                weights = weights.unsqueeze(0)
            x = torch.sum(x * weights,dim = -1,keepdim = True)
        return x
    def separete_save(self,path,name):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.net.state_dict(),path + f"/{name}_block.pt")
        torch.save(self.output_layer.state_dict(),path + f"/{name}_outlayer.pt")

    def load(self,path,name):
        self.net.load_state_dict(torch.load(path + f"/{name}_block.pt"))
        self.output_layer.load_state_dict(torch.load(path + f"/{name}_outlayer.pt"))
        if self.use_mas:
            self.net._set_parameter()
    def load_expert(self,path,name):
        if not self.use_mas:
            return 
        expert_block_state_dict = torch.load(path + f"/{name}_block.pt")
        self.net.load_expert_state_dict(expert_block_state_dict)
        self.output_layer.load_state_dict(torch.load(path + f"/{name}_outlayer.pt"))
        self.net._set_parameter()

class VControlNet_v2(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_size = [256,256],use_mas = False,extra_input_dim = 0,allow_retrain = False):
        super().__init__()
        self.use_mas = use_mas
        self.extra_intput_dim = extra_input_dim
        if self.use_mas:
            self.net = ControlBlock(input_dim,hidden_size[-1],extra_input_dim = extra_input_dim,hidden_size=hidden_size,allow_retrain = allow_retrain)
        else:
            self.net = Block(input_dim,hidden_size[-1],hidden_size = hidden_size)
        self.output_dim = output_dim
        self.output_layers = nn.ModuleList()
        for _ in range(output_dim):
            self.output_layers.append(nn.Linear(hidden_size[-1],1))
    def _multi_head(self,x):
        output = []
        for i in range(self.output_dim):
            output.append(self.output_layers[i](x))
        output = torch.cat(output,dim = -1)
        return output
    def forward(self,x,extra_input = None,weights = None):
        if self.use_mas:
            x = self.net.forward(x,extra_input)
        else:
            x = self.net.forward(x)
        x = self._multi_head(x)
        if weights is not None:
            ## x.shape (bz,output_dim)
            ## weights (output_dim)
            if weights.ndim == 1:
                weights = weights.unsqueeze(0)
            x = torch.sum(x * weights,dim = -1,keepdim = True)
        return x
    def separete_save(self,path,name):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.net.state_dict(),path + f"/{name}_block.pt")
        torch.save(self.output_layers.state_dict(),path + f"/{name}_outlayer.pt")

    def load(self,path,name):
        self.net.load_state_dict(torch.load(path + f"/{name}_block.pt"))
        self.output_layers.load_state_dict(torch.load(path + f"/{name}_outlayer.pt"))
        if self.use_mas:
            self.net._set_parameter()
    def load_expert(self,path,name):
        if not self.use_mas:
            return 
        expert_block_state_dict = torch.load(path + f"/{name}_block.pt")
        self.net.load_expert_state_dict(expert_block_state_dict)
        self.output_layers.load_state_dict(torch.load(path + f"/{name}_outlayer.pt"))
        self.net._set_parameter()



class VExpert(object):
    def __init__(self,obs_dim,act_dim,reward_dim,hidden_size = [256,256]):
        obs_dim,act_dim,reward_dim = int(obs_dim),int(act_dim),int(reward_dim)
        self.critic = VControlNet(obs_dim,reward_dim,hidden_size = hidden_size,use_mas = False,)
        self.actor = ControlNet(obs_dim,act_dim,hidden_size = hidden_size,use_mas = False)
    
    def get_value(self,x,style,weights = None):
        return self.critic(x,style,weights = weights)
    def get_action_and_value(self,x,style,action = None,weights = None):
        logits = self.actor(x,style)
        probs = Categorical(logits = logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action),probs.entropy(), self.critic(x,style,weights = weights)

    def save(self,path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.critic.separete_save(path,"critic")
        self.actor.separete_save(path,"actor")
    def load(self,path):
        assert os.path.exists(path)
        self.critic.load(path,"critic")
        self.actor.load(path,"actor")
    def to(self,device):
        self.critic.to(device)
        self.actor.to(device)
        return self
    

class VStyleExpert(object):
    def __init__(self,obs_dim,act_dim,reward_dim,style_dim,allow_retrain = False,hidden_size = [256,256]):
        obs_dim,act_dim,reward_dim = int(obs_dim),int(act_dim),int(reward_dim)
        self.critic = VControlNet(obs_dim,reward_dim,hidden_size = hidden_size,
                                  use_mas = True,
                                  extra_input_dim = style_dim,
                                  allow_retrain = allow_retrain)
        self.actor = ControlNet(obs_dim,act_dim,hidden_size = hidden_size,use_mas = True,
                                extra_input_dim = style_dim,allow_retrain=allow_retrain)
    
    def get_value(self,x,style,weights = None):
        return self.critic(x,style,weights = weights)
    def get_action_and_value(self,x,style,action = None,weights = None):
        logits = self.actor(x,style)
        probs = Categorical(logits = logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action),probs.entropy(), self.critic(x,style,weights = weights)
    def load_expert(self,path):
        if os.path.exists(path):
            self.critic.load_expert(path,"critic")
            self.actor.load_expert(path,"actor")
            print(f"Model Loaded From: {path}") 
        else:
            print("Path Not Exist") 
    def save(self,path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.critic.separete_save(path,"critic")
        self.actor.separete_save(path,"actor")
    def load(self,path):
        assert os.path.exists(path)
        self.critic.load(path,"critic")
        self.actor.load(path,"actor")
    def to(self,device):
        self.critic.to(device)
        self.actor.to(device)
        return self

class VMLPStyleExpert(object):
    def __init__(self,obs_dim,act_dim,reward_dim,style_dim,allow_retrain = False,hidden_size = [256,256]):
        obs_dim,act_dim,reward_dim = int(obs_dim),int(act_dim),int(reward_dim)
        self.critic = VMLPNet(obs_dim,reward_dim,hidden_size,extra_input_dim=style_dim)
        self.actor = MLPNet(obs_dim,act_dim,hidden_size,extra_input_dim=style_dim)
    
    def get_value(self,x,style,weights = None):
        return self.critic(x,style,weights = weights)
    def get_action_and_value(self,x,style,action = None,weights = None):
        logits = self.actor(x,style)
        probs = Categorical(logits = logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action),probs.entropy(), self.critic(x,style,weights = weights)
    
    def save(self,path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.critic.separete_save(path,"critic")
        self.actor.separete_save(path,"actor")
    def load(self,path):
        assert os.path.exists(path)
        self.critic.load(path,"critic")
        self.actor.load(path,"actor")
    def to(self,device):
        self.critic.to(device)
        self.actor.to(device)
        return self