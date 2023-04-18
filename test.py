from algorithm.ppo import Learner
from configs.args_parser import Parameters
from configs import Factor_dictionary

if __name__ == '__main__':
    # env_id = "deep-sea-treasure-v0"
    env_id = 'mo-lunar-lander-v2'
    para = Parameters(env_id=env_id,config_path=None)
    learner = Learner(para,env_id=env_id,factors=Factor_dictionary[env_id])
    learner.agent.load_expert(path="logs/mo-lunar-lander-v2_Expert")
    learner.learn(para.total_timesteps,para.num_steps)
    learner.save()
    learner.load()