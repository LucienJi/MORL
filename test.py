import json 
from algorithm.sampler.buffer import TrainingSet
from algorithm.learner.ppo_learner import PPO_Learner
from algorithm.sampler.Coordinator import Coordinator
from algorithm.logger.Logger import BasicLogger
from algorithm.evaluator.Evaluator import Evaluator
import ray 

if __name__ == '__main__':
    with open("configs/toy_configs.json","r") as f:
        configs = json.load(f)
    logger = BasicLogger(configs)
    buffer = TrainingSet(configs)
    learner = PPO_Learner(configs,buffer,logger)
    learner.save_model()
    sampler = Coordinator(configs,buffer,eval_mode = False,logger=logger)
    evaluator = Evaluator(configs,logger)
    task_ids,weight_list,style_list = evaluator.evaluate_all_tasks()
    sampler.set_tasks(task_ids,weight_list,style_list)
    for i in range(50):
        sampler.collect_training_data()
        learner.learn()
        learner.save_model()
        sampler.fetch_model()
        buffer.clear()
        logger.dump_summary()
        print("Epoch: ",i)
    task_ids,weight_list,style_list = evaluator.evaluate_all_tasks()
    eval_res = sampler.evaluate_tasks(task_ids,weight_list,style_list)
    eval_res = evaluator.calc_statistics(eval_res)
        
    #! Eval
    