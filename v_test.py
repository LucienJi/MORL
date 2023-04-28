from algorithm.v_ppo import VLearner
from configs.args_parser import Parameters
from configs import Factor_dictionary
from algorithm.analysis import MO_Analysis
if __name__ == '__main__':
    env_id =  'deep-sea-treasure-v0'
    para = Parameters(env_id=env_id,config_path=None)
    learner = VLearner(para,env_id,Factor_dictionary[env_id],'controlnet')
    if hasattr(learner.agent, 'load_expert'):
        learner.agent.load_expert(path="logs/deep-sea-treasure-v0_V_Expert")
    learner.learn(para.total_timesteps,para.num_steps)
    learner.save()