from algorithm.v_ppo import VLearner
from configs.args_parser import Parameters
from configs import Factor_dictionary
from algorithm.analysis import MO_Analysis
if __name__ == '__main__':
    # env_id =  'deep-sea-treasure-v0'
    env_id = 'mo-halfcheetah-v4'
    para = Parameters(env_id=env_id,config_path=None)
    learner = VLearner(para,env_id,Factor_dictionary[env_id],'expert')
    if hasattr(learner.agent, 'load_expert'):
        learner.agent.load_expert(path="logs/deep-sea-treasure-v0_V_Expert_v2")
    learner.learn(para.total_timesteps,para.num_steps)
    learner.save()
    # learner.sampler._save_debugs(para.save_path)