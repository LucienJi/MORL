import mo_gymnasium as mo_gym
import gym 
import numpy as np
import random


def make_styled_env(env_id,seed,reward_factors):
    assert type(reward_factors) == list 
    def thunk():
        env = mo_gym.make(env_id)
        env = mo_gym.LinearReward(env, weight=np.array(reward_factors))
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = mo_gym.MORecordEpisodeStatistics(env)
        if hasattr(env, 'seed'):
            env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def generate_reward_factors(factors:dict,seed = 0):
    random.seed(seed)
    reward_factors = {}
    reward_factors_list = []
    style_state = []
    for factor_name, factor_info in factors.items():
        if factor_info['fine_grit'] > 0:
            if factor_info['lower_bound'] == factor_info['upper_bound']:
                reward_factors[factor_name] = factor_info['lower_bound']
                reward_factors_list.append(factor_info['lower_bound'])
            else:
                grit_factor = np.linspace(
                    factor_info['lower_bound'],
                    factor_info['upper_bound'],
                    int((factor_info['upper_bound'] - factor_info['lower_bound']) / factor_info['fine_grit']) + 1
                    )
                reward_factors[factor_name] = random.choice(grit_factor)
                reward_factors_list.append(reward_factors[factor_name])
        else:
            reward_factors[factor_name] = \
                random.uniform(factor_info['lower_bound'], factor_info['upper_bound'])
            reward_factors_list.append(reward_factors[factor_name])

        if factor_info['lower_bound'] == factor_info['upper_bound']:
            style_value= factor_info['upper_bound'] / 2
        else:
            style_value = (reward_factors[factor_name] - factor_info['lower_bound']) / \
                                (factor_info['upper_bound'] - factor_info['lower_bound'])
                
        style_state.append(style_value)
    return reward_factors_list, reward_factors,style_state