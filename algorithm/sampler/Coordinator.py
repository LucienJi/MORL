from algorithm.utils import make_styled_env,generate_reward_factors
from algorithm.sampler.Worker import Worker,RemoteWorker
from algorithm.sampler.ppo_sampler import PPO_Agent
from algorithm.sampler.buffer import TrainingSet
import ray 
from algorithm.logger.Logger import BasicLogger

class Coordinator(object):
    def __init__(self,configs,buffer:TrainingSet,eval_model = False,
                 logger:BasicLogger=None) -> None:
        self.eval_model = eval_model
        if self.eval_model:
            self.model_path = configs["eval_model_path"]
        else:
            self.model_path = configs["model_path"]
        
        self.num_workers = configs["num_workers"]   
        self.use_remote = configs["use_remote"]
        self.name = configs['name']
        self.env_id = configs["env_id"]
        self.reward_factor = configs["reward_factor"]
        #! Data Buffer 
        self.buffer = buffer

        #! init agents
        if self.use_remote:
            self.workers = [RemoteWorker.remote(configs,PPO_Agent,i) for i in range(self.num_workers)]
        else:
            self.workers = [Worker(configs,PPO_Agent,i) for i in range(self.num_workers)]
        self.fetch_model()

        self.logger = logger 
    
    def set_tasks(self,task_id,weight_list,style_list):
        if self.use_remote:
            ray.get([worker.get_tasks.remote(task_id,weight_list,style_list) for worker in self.workers])
        else:
            [worker.get_tasks(task_id,weight_list,style_list) for worker in self.workers]

    def fetch_model(self):
        if self.use_remote:
            [worker.fetch_model.remote(self.model_path,self.name) for worker in self.workers]
        else:
            [worker.fetch_model(self.model_path,self.name) for worker in self.workers]
    
    def collect_training_data(self):
        num_samples = 0
        if self.use_remote:
            data_list = ray.get([worker.sample_one_traj.remote() for worker in self.workers])
        else:
            data_list = [worker.sample_one_traj() for worker in self.workers]
        for i,data in enumerate(data_list):
            memory,statistics = data[0],data[1]
            self.buffer.append_instance(memory)
            num_samples += len(memory)

            if self.logger is not None:
                self.logger.log_detail(statistics)
        

