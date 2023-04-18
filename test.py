from algorithm.ppo import Learner
from configs.args_parser import Parameters
from configs import Factor_dictionary

if __name__ == '__main__':
    env_id = "deep-sea-treasure-v0"
    para = Parameters(config_path= None,save_path = "configs/config")
    learner = Learner(para,env_id=env_id,factors=Factor_dictionary[env_id],allow_retrain=True)
    learner.learn(para.total_timesteps,para.num_steps)