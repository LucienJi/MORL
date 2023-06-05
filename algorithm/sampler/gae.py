import numpy as np
import torch as th


class GAE(object):
    def __init__(self):
        pass

    def estimate_advantages(self, rewards, masks, values, gamma, tau, bootstrap_value):
        deltas = [1 for i in range(len(rewards))]
        advantages = [1 for i in range(len(rewards))]
        returns = [1 for i in range(len(rewards))]
        prev_value = bootstrap_value
        prev_advantage = 0
        for i in reversed(range(len(rewards))):
            deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
            advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]
            prev_value = values[i]
            prev_advantage = advantages[i]
            returns[i] = values[i] + advantages[i]

        return advantages, returns
    
    def estimate_advantages_vec(self,reward_vec,mask,values,gamma,tau,bootstrap_value):
        """
        reward_vec: (style_dim)
        bootstrap_value: (style_dim)
        values: (style_dim)
        mask: bool
        """
        deltas = [1 for i in range(len(reward_vec))]
        advantages = [1 for i in range(len(reward_vec))]
        returns = [1 for i in range(len(reward_vec))]
        prev_value = bootstrap_value
        prev_advantage = np.zeros_like(bootstrap_value)
        for i in reversed(range(len(reward_vec))):
            deltas[i] = reward_vec[i] + gamma * prev_value * mask[i] - values[i]
            advantages[i] = deltas[i] + gamma * tau * prev_advantage * mask[i]
            prev_value = values[i]
            prev_advantage = advantages[i]
            returns[i] = values[i] + advantages[i]
        return advantages,returns
