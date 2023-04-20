import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import mo_gymnasium as mo_gym
import gymnasium as gym
from  rlmodel.controlnet import StyleExpert,Expert,MLPStyleExpert
from algorithm.utils import make_styled_env,generate_reward_factors
from .utils import MO_Stats
from torch.utils.tensorboard import SummaryWriter

class MO_Analysis(object):
    def __init__(self,args,env_id,factors:dict):
        self.args = args
        self.env_id = env_id
        self.env = mo_gym.make(env_id)
        self.obs_dim,self.act_dim = self.env.observation_space.shape[0],self.env.action_space.n
        self.factors = factors #! dictionary, predefine the bounds of each factor
        self.style_dim = len(self.factors.keys())
        if self.args.use_mas:
            print("################### Using MAS ###################")
            # self.agent = StyleExpert(self.obs_dim,self.act_dim,self.style_dim,allow_retrain=self.args.allow_retrain)
            self.agent = MLPStyleExpert(self.obs_dim,self.act_dim,self.style_dim)
        else:
            print("################### Using Expert ###################")
            self.agent = Expert(self.obs_dim,self.act_dim)
        self.use_mas = self.args.use_mas
        self.set_style_state()
    def set_style_state(self,style_state = None):
        if style_state is None:
            reward_factor_list,reward_factor,style_state = generate_reward_factors(self.factors,self.args.seed)
        self.style_state_np = np.array(style_state)
        self.style_state = torch.from_numpy(self.style_state_np).float().unsqueeze(0)
    def _init_track(self,writer = None):
        if writer is not None:
            self.writer = writer
            self.track = True
        else:
            run_name = f"{self.env_id}__{self.args.exp_name}"
            if self.args.track:
                import wandb
                wandb.init(
                    project=self.args.wandb_project_name,
                    entity=self.args.wandb_entity,
                    sync_tensorboard=True,
                    config=vars(self.args),
                    name=run_name,
                    monitor_gym=True,
                    save_code=False,
                )
            self.writer = SummaryWriter(f"{self.args.save_path}/eval")
            self.writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
            )


    def evaluate(self,min_samples,min_ep):
        mo_analysis = MO_Stats()
        step_ct = 0
        ep_ct = 0 
        env = mo_gym.make(self.env_id)
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        while step_ct < min_samples and ep_ct < min_ep:
            obs,info = env.reset()
            done = False
            while not done and step_ct < min_samples:
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(torch.from_numpy(obs).float().unsqueeze(0),
                                                                                self.style_state)
                next_obs, reward, done, truncted,info = env.step(action.numpy()[0])
                mo_analysis.update(obs,action,reward)
                obs = next_obs
                step_ct += 1
            ep_ct += 1
        return mo_analysis.raw_data()

    def mas_eval(self,min_samples,min_ep,style_state,writer = None):
        self._init_track(writer)
        if type(style_state) != list:
            style_state = [style_state]
        for s_state in style_state:
            self.set_style_state(s_state)
            
            actions,states,rewards = self.evaluate(min_samples,min_ep)
            for i in range(actions.shape[1]):
                self.writer.add_histogram(f"{str(self.style_state_np)}/action",actions[:,i],i,bins=self.env.action_space.n)
            for i in range(states.shape[1]):
                self.writer.add_histogram(f"{str(self.style_state_np)}/states",states[:,i],i)
            for i in range(rewards.shape[1]):
                self.writer.add_histogram(f"{str(self.style_state_np)}/reward",rewards[:,i],i)
        self.writer.close()
    def save(self, save_path = None):
        if save_path is None:
            self.agent.save(self.args.model_path)
        else:
            self.agent.save(save_path)
    def load(self, load_path = None):
        if load_path is None:
            self.agent.load(self.args.model_path)
        else:
            self.agent.load(load_path)