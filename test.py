import json 
from algorithm.sampler.buffer import TrainingSet
from algorithm.learner.ppo_learner import PPO_Learner
from algorithm.sampler.Coordinator import Coordinator
from algorithm.logger.Logger import BasicLogger
import ray 

if __name__ == '__main__':
    with open("configs/toy_configs.json","r") as f:
        configs = json.load(f)
    logger = BasicLogger(configs)
    buffer = TrainingSet(configs)
    learner = PPO_Learner(configs,buffer,logger)
    learner.save_model()
    sampler = Coordinator(configs,buffer,eval_model = False,logger=logger)
    for _ in range(10):
        sampler.collect_training_data()
        learner.learn()
        learner.save_model()
        sampler.fetch_model()
        buffer.clear()
        logger.dump_summary()