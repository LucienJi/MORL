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


    #! n_epochs = 50
    n_epochs = 200
    total_samples = configs['num_workers'] * configs['max_traj_len'] * n_epochs
    print("Total Samples: ",total_samples)
    for i in range(n_epochs):
        sampler.collect_training_data()
        print("Epoch: ",i, "Buffer Size: ",buffer.len())
        learner.learn()
        learner.save_model()
        sampler.fetch_model()
        buffer.clear()
        logger.dump_summary()
        
    task_ids,weight_list,style_list = evaluator.evaluate_all_tasks()
    eval_res = sampler.evaluate_tasks(task_ids,weight_list,style_list,n_traj=10)
    eval_res = evaluator.calc_statistics(eval_res)
        
    #! Eval
    