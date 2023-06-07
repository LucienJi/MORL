import json 
from algorithm.sampler.buffer import TrainingSet
from algorithm.learner.ppo_learner import PPO_Learner
from algorithm.sampler.Coordinator import Coordinator
import ray 

if __name__ == '__main__':
    with open("configs/toy_configs.json","r") as f:
        configs = json.load(f)

    buffer = TrainingSet(configs)
    learner = PPO_Learner(configs,buffer)
    learner.save_model()
    sampler = Coordinator(configs,buffer,eval_model = False)
    print(buffer.len())
    sampler.collect_training_data()
    print(buffer.len())
    learner.learn()
