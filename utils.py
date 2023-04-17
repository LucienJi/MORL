import random 
import numpy as np 
import torch    
def get_style_state(factors,seed = 0):
    random.seed(seed)
    reward_factors = {}
    style_state = []
    for factor_name, factor_info in factors.items():
        if factor_info['fine_grit'] > 0:
            if factor_info['lower_bound'] == factor_info['upper_bound']:
                reward_factors[factor_name] = factor_info['lower_bound']
            else:
                grit_factor = np.linspace(
                    factor_info['lower_bound'],
                    factor_info['upper_bound'],
                    int((factor_info['upper_bound'] - factor_info['lower_bound']) / factor_info['fine_grit']) + 1
                )
                reward_factors[factor_name] = random.choice(grit_factor)
        else:
            reward_factors[factor_name] = random.uniform(factor_info['lower_bound'], factor_info['upper_bound'])

        if factor_info['lower_bound'] == factor_info['upper_bound']:
            style_value = factor_info['upper_bound'] / 2
        else:
            style_value = (reward_factors[factor_name] - factor_info['lower_bound']) / \
                            (factor_info['upper_bound'] - factor_info['lower_bound'])
        style_state.append(style_value)

    return style_state, reward_factors



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer



