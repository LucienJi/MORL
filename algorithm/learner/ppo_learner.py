from algorithm.learner.models.mas_policy import PolicyNet
from algorithm.learner.model_utils import deserialize_model,serialize_model
import torch as th
import mo_gymnasium as mo_gym
import numpy as np 
from algorithm.learner.model_utils import to_device
from algorithm.sampler.buffer import Instance,TrainingSet
import os
from algorithm.logger.Logger import BasicLogger 
class PPO_Learner:
    def __init__(self,configs,buffer:TrainingSet,logger:BasicLogger=None) -> None:
        #! Parameter Part 
        self.clip_epsilon = configs['ppo_clip']
        self.dual_clip_c = configs['dual_clip_c']
        self.ent_coef = configs['ent']
        self.lr = configs['lr']
        self.grad_clip = configs['grad_clip']
        self.batch_size = configs['batch_size']
        self.model_path = configs['model_path']
        self.name = configs['name']
        self.local_rank = 0
        
        self.env_id = configs['env_id']    
        
        self.env = mo_gym.make(self.env_id)
        self.env = mo_gym.LinearReward(self.env)
        self.obs_dim,self.act_dim = self.env.observation_space.shape[0],self.env.action_space.shape[0]
        self.reward_factor = configs["reward_factor"]  
        self.style_dim = len(self.reward_factor.keys())

        #！ Model Part 
        self.net = PolicyNet(self.obs_dim,self.act_dim,self.style_dim)
        self.optimizer = th.optim.Adam(self.net.parameters(),lr = self.lr)

        if configs['load_model']:
            self.load_model()
        if configs['device'] == 'cuda':
            self.net.cuda()
        self.device = configs['device']

        #! Date Part
        self.training_set = buffer
        
        self.logger = logger
    
    def save_model(self,path = None):
        if path is None:
            path = self.model_path
        if not os.path.exists(path):
            os.makedirs(path)
        serialize_model(self.net,path,self.name)
    
    def load_model(self,path = None):
        if path is None:
            path = self.model_path
        if os.path.exists(os.path.join(path,self.name)):
            self.net = deserialize_model(path,self.name)
    
    def update(self,training_batch):
        """
        training_batch: dict of torch.tensor
        """
        states = training_batch["states"]
        styles = training_batch["styles"]
        reward_weights = training_batch["reward_weights"]
        actions = training_batch["actions"]
        q_values = training_batch["q_values"] # vector
        ad_std = th.std(training_batch["advantages"],dim = 0,keepdim=True)
        advantages_cpu = training_batch["advantages"] / ad_std
        advantages = advantages_cpu
        advantages_scalar = (advantages * reward_weights).sum(-1)
        fixed_log_probs = training_batch["old_log_prob"]

        mean,logstd,value_pred = self.net(states,styles)
        ## Value Loss 
        value_loss = (value_pred - q_values).pow(2).mean()

        ## Policy Loss 
        probs = th.distributions.Normal(mean,logstd.exp())
        new_log_probs = probs.log_prob(actions).sum(-1,keepdim=True)
        ratio = (new_log_probs - fixed_log_probs).exp()
        surr1 = ratio * advantages_scalar
        surr2 = th.clamp(ratio,1.0 - self.clip_epsilon,1.0 + self.clip_epsilon) * advantages_scalar
        surr = th.min(surr1,surr2)
        surr3 = th.min(self.dual_clip_c * advantages_scalar,th.zeros_like(advantages_scalar))
        surr = th.max(surr,surr3)
        psurr = -surr.mean()

        ## Entropy Loss
        entropy = probs.entropy().mean()

        with th.no_grad():
            log_ratio = new_log_probs - fixed_log_probs
            approx_kl1 = th.mean(-log_ratio).cpu().numpy()
            approx_kl2 = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
            clipfrac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
        
        policy_surr =  psurr + 0.5 * value_loss - self.ent_coef * entropy
        self.optimizer.zero_grad()
        policy_surr.backward()
        th.nn.utils.clip_grad_norm_(self.net.parameters(),self.grad_clip)
        self.optimizer.step()

        info = {
            "Learner/policy_loss":psurr.item(),
            "Learner/value_loss":value_loss.item(),
            "Learner/entropy":entropy.item(),
            "Learner/approx_kl1":approx_kl1,
            "Learner/approx_kl2":approx_kl2,
            "Learner/clipfrac":clipfrac,
            "Learner/adv_mean":advantages_scalar.mean().item(),
            "Learner/adv_std":advantages_scalar.std().item(),
        }
        vec_info = {}
        for i in range(self.style_dim):
            vec_info[f"Learner/adv_{i}"] = advantages_cpu[:,i].mean().item()
            vec_info[f"Learner/adv_{i}_std"] = advantages_cpu[:,i].std().item()
            vec_info[f'Learner/q_value_{i}'] = q_values[:,i].mean().item()
        
        info.update(vec_info)
        return info 


    def learn(self,):
        """
        data: dict of np.ndarray collected from sampler 
        """
        bz = self.training_set.len()
        minibatch_size = min(self.batch_size,bz)
        for index in self.training_set.get_training_index(minibatch_size):
            training_batch = self.training_set.slice(index)
            training_batch = to_device(training_batch,device = "cpu")
            info = self.update(training_batch)
            if self.logger is not None:
                self.logger.log_detail(info)
        

        

        

