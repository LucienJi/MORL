from algorithm.ppo import Learner
from configs.args_parser import Parameters
from configs import Factor_dictionary
from algorithm.analysis import MO_Analysis
if __name__ == '__main__':
    env_id = "deep-sea-treasure-v0"
    # env_id = 'mo-lunar-lander-v2'
    # env_id = 'minecart-v0'
    para = Parameters(env_id=env_id,config_path=None)
    learner = Learner(para,env_id=env_id,factors=Factor_dictionary[env_id])
    if hasattr(learner.agent, 'load_expert'):
        learner.agent.load_expert(path="logs/deep-sea-treasure-v0_expert")
    learner.learn(para.total_timesteps,para.num_steps)
    learner.save()

    mo_analysis = MO_Analysis(para,env_id,Factor_dictionary[env_id])
    mo_analysis.load()
    # style_states = [
    #     [0.5,0.5,0.0,0.5],
    #     [0.5,0.5,0.5,0.5],
    #     [0.5,0.5,1.0,0.5],
    # ]
    style_states = []
    for i in range(10):
        tmp = [0.5,0.0 + i*0.1]
        style_states.append(tmp)

    if hasattr(learner, 'writer'):
        mo_analysis.mas_eval(5000,10,style_states,learner.writer)  
    else:
        mo_analysis.mas_eval(5000,10,style_states,None)  