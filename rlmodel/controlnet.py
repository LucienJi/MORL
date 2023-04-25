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
    """
    trainable block and locked block
    """
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
        self.fc_linear = nn.Sequential(*fc_linear)
        self.last_dim = hidden_size[-1]
        self.last_layer = nn.Linear(hidden_size[-1],output_dim)
    
    def forward(self,x):
        x = self.fc_linear(x)
        x = self.last_layer(x)
        return x

class ControlBlock(nn.Module):
    """
    contain two block, one is locked, one is trainable
    contain two zero layer
    """
    def __init__(self,input_dim,output_dim,extra_input_dim,hidden_size,allow_retrain = False):
        super().__init__()
        #! trainable and locked block must has the same size 
        self.trainable_block = Block(input_dim,output_dim,hidden_size)
        self.locked_block = Block(input_dim,output_dim,hidden_size)

        ##TODO 改动1： zero layer 需要 state or style 的信息
        self.zero_layer1 = nn.Linear(extra_input_dim + input_dim,input_dim)
        # self.zero_layer1 = nn.Sequential(
        #     nn.Linear(extra_input_dim + input_dim,hidden_size[0]),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_size[0],input_dim)
        # )
        self.zero_layer2 = nn.Linear(output_dim + extra_input_dim,output_dim)
        # self.zero_layer2 = nn.Sequential(
        #     nn.Linear(output_dim + extra_input_dim,hidden_size[0]),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_size[0],output_dim)
        # )
        self.allow_retrain = allow_retrain
        self.init()
        self._set_parameter()
    
    def _set_parameter(self):
        for p in self.trainable_block.parameters():
            p.requires_grad = True
        for p in self.locked_block.parameters():
            p.requires_grad = self.allow_retrain

    def init(self):
        for m in self.modules():
            if m in (self.zero_layer1,self.zero_layer2):
                constant_init(m,val=0.0)

    def forward(self,x,extra_input):
        input1 = torch.cat([x,extra_input],dim = 1)
        delta_x = self.zero_layer1(input1)
        x_ = x + delta_x
        y_ = self.trainable_block.forward(x_)
        input2 = torch.cat([y_,extra_input],dim = 1)
        delta_y = self.zero_layer2(input2)
        return self.trainable_block(x) + delta_y
    # def forward(self,x,extra_input):
    #     delta_x = self.zero_layer1(extra_input)
    #     x_ = x + delta_x
    #     delta_y = self.zero_layer2(self.trainable_block.forward(x_))
    #     y_ = self.locked_block.forward(x) + delta_y
    #     return y_
    
    def load_expert_state_dict(self,state_dict):
        self.locked_block.load_state_dict(state_dict)
        self.trainable_block.load_state_dict(state_dict)
        with torch.no_grad():
            for p in self.trainable_block.parameters():
                p.requires_grad = True
        with torch.no_grad():
            for p in self.locked_block.parameters():
                p.requires_grad = self.allow_retrain 
class MLPNet(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_size = [256,256],extra_input_dim = 0):
        super().__init__()
        self.net = Block(input_dim= input_dim + extra_input_dim,output_dim = hidden_size[-1],hidden_size = hidden_size)
        self.output_layer = nn.Linear(hidden_size[-1],output_dim)
    def forward(self,x,extra_input = None):
        if extra_input is not None:
            x = torch.cat([x,extra_input],dim = 1)
        x = self.net.forward(x)
        x = self.output_layer(x)
        return x
    
    def separete_save(self,path,name):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.net.state_dict(),path + f"/{name}_block.pt")
        torch.save(self.output_layer.state_dict(),path + f"/{name}_outlayer.pt")

    def load(self,path,name):
        self.net.load_state_dict(torch.load(path + f"/{name}_block.pt"))
        self.output_layer.load_state_dict(torch.load(path + f"/{name}_outlayer.pt"))

class ControlNet(nn.Module):
    #! 存储的时候，state dict 分为 net 和 outlayer，，load expert state dict 的时候需要将 net 的 state dict 分给两个 block
    def __init__(self,input_dim,output_dim,hidden_size = [256,256],use_mas = False,extra_input_dim = 0,allow_retrain = False):
        super().__init__()
        self.use_mas = use_mas
        self.extra_intput_dim = extra_input_dim
        if self.use_mas:
            self.net = ControlBlock(input_dim,hidden_size[-1],extra_input_dim = extra_input_dim,hidden_size=hidden_size,allow_retrain = allow_retrain)
        else:
            self.net = Block(input_dim,hidden_size[-1],hidden_size = hidden_size)
        self.output_layer = nn.Linear(hidden_size[-1],output_dim) # 这一部分也是 trainable 的
    def forward(self,x,extra_input = None):
        if self.use_mas:
            x = self.net.forward(x,extra_input)
        else:
            x = self.net.forward(x)
        x = self.output_layer(x)
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


class Expert(object):
    def __init__(self,obs_dim,act_dim,hidden_size = [256,256]) -> None:
        obs_dim,act_dim = obs_dim,act_dim
        self.critic = ControlNet(obs_dim,1,hidden_size=hidden_size,use_mas = False,extra_input_dim = 0,allow_retrain = True)
        self.actor = ControlNet(obs_dim,act_dim,hidden_size=hidden_size,use_mas = False,extra_input_dim = 0,allow_retrain = True)
    def get_value(self, x,style):
        return self.critic(x)

    def get_action_and_value(self, x,style, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
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


class StyleExpert(object):
    def __init__(self,obs_dim,act_dim,style_dim,allow_retrain,hidden_size =[256,256]) -> None:
        obs_dim,act_dim = obs_dim,act_dim
        self.critic = ControlNet(obs_dim,1,hidden_size=hidden_size,use_mas = True,extra_input_dim = style_dim,allow_retrain = allow_retrain)
        self.actor = ControlNet(obs_dim,act_dim,hidden_size=hidden_size,use_mas = True,extra_input_dim = style_dim,allow_retrain = allow_retrain)
        
    def to(self,device):
        self.actor.to(device)
        self.critic.to(device)
        return self
    def get_value(self, x,style):
        return self.critic(x,style)

    def get_action_and_value(self, x,style, action=None):
        logits = self.actor(x,style)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x,style)
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

class MLPStyleExpert(object):
    def __init__(self,obs_dim,act_dim,style_dim) -> None:
        obs_dim,act_dim = obs_dim,act_dim
        self.critic = MLPNet(obs_dim,1,hidden_size=[256,256],extra_input_dim = style_dim)
        self.actor = MLPNet(obs_dim,act_dim,hidden_size=[256,256],extra_input_dim = style_dim)
        
    def to(self,device):
        self.actor.to(device)
        self.critic.to(device)
        return self
    def get_value(self, x,style):
        return self.critic(x,style)

    def get_action_and_value(self, x,style, action=None):
        logits = self.actor(x,style)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x,style)

    def save(self,path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.critic.separete_save(path,"critic")
        self.actor.separete_save(path,"actor")
    def load(self,path):
        assert os.path.exists(path)
        self.critic.load(path,"critic")
        self.actor.load(path,"actor")






