import os 
import json 
import wandb
from algorithm.logger.utils import setup_logger,create_path
import time 
import numpy as np 
from torch.utils.tensorboard import SummaryWriter

class SummaryLogger:
    def __init__(self,configs,summary_file_to_write) -> None:
        self.configs = configs
        self.use_wandb = configs['use_wandb']
        # self.summary_writer = tf.summary.create_file_writer(summary_file_to_write)
        self.summary_writer = SummaryWriter(summary_file_to_write)
        if self.configs['use_wandb']:
            wandb.init(
                project="MAS",
                entity='jingtianji',
                sync_tensorboard=True,
                config=self.configs,
                name=self.configs['name'],
                monitor_gym=False,
                save_code=False,
            )

        self.tag_values_dict = {}
        self.tag_step_dict = {} #! 这是记录dump的次数
        self.tag_output_threshold_dict = {}
        self.tag_func_dict = {}
        self.tag_total_add_count = {} #！ 这是记录add的次数
        self.total_tag_type = [
            "time_avg", "time_total",
            "avg", "total", "max", "min"
        ]
        self.tag_bins = {}

    def add_tag(self, tag, output_threshold = 0, cal_type = 'avg', time_threshold=0, bins=100):
        self.tag_values_dict[tag] = []
        self.tag_step_dict[tag] = 0
        self.tag_output_threshold_dict[tag] = output_threshold
        self.tag_func_dict[tag] = cal_type
        self.tag_total_add_count[tag] = 0
        if cal_type.find("histogram") != -1:
            self.tag_bins[tag] = bins
    
    def add_info(self,tag,value):
        if tag not in self.tag_values_dict.keys():
            #! init tag
            self.add_tag(tag,output_threshold=0,cal_type='avg')
        if type(value) == list:
            self.tag_values_dict[tag].extend(value)
        else:
            self.tag_values_dict[tag].append(value)
        self.tag_total_add_count[tag] += 1
    
    def dump_summary(self):
        for tag in self.tag_values_dict.keys():
            if len(self.tag_values_dict[tag]) < self.tag_output_threshold_dict[tag]: continue
            if self.tag_func_dict[tag].find("histogram") != -1:
                # each value is a list
                all_values = []
                for i in self.tag_values_dict[tag]:
                    all_values.extend(i)
                self.summary_writer.add_histogram(tag, all_values, global_step=self.tag_step_dict[tag])
            else:
                if self.tag_func_dict[tag] == "avg":
                    avg_value = sum(self.tag_values_dict[tag]) / len(self.tag_values_dict[tag])
                elif self.tag_func_dict[tag] == "total":
                    avg_value = sum(self.tag_values_dict[tag])
                elif self.tag_func_dict[tag] == "max":
                    avg_value = max(self.tag_values_dict[tag])
                elif self.tag_func_dict[tag] == "min":
                    avg_value = min(self.tag_values_dict[tag])
                elif self.tag_func_dict[tag] == "sd":
                    avg_value = np.array(self.tag_values_dict[tag]).std()

                #! 暂时不能处理 vector
                self.summary_writer.add_scalar(tag, avg_value, global_step=self.tag_step_dict[tag])
            self.tag_step_dict[tag] += 1
            self.tag_values_dict[tag] = []


    def add_summary(self,tag,value,timestamp=time.time()):
        if tag not in self.tag_values_dict.keys():
            #! init tag
            self.add_tag(tag,output_threshold=0,cal_type='avg')
        
        self.tag_values_dict[tag].append(value)
        self.tag_total_add_count[tag] += 1  
        if len(self.tag_values_dict[tag]) >= self.tag_output_threshold_dict[tag]:
            if self.tag_func_dict[tag].find("histogram") != -1:
                # each value is a list
                all_values = []
                for i in self.tag_values_dict[tag]:
                    all_values.extend(i)
                self.summary_writer.add_histogram(tag, all_values, global_step=self.tag_step_dict[tag])
            else:
                if self.tag_func_dict[tag] == "avg":
                    avg_value = sum(self.tag_values_dict[tag]) / len(self.tag_values_dict[tag])
                elif self.tag_func_dict[tag] == "total":
                    avg_value = sum(self.tag_values_dict[tag])
                elif self.tag_func_dict[tag] == "max":
                    avg_value = max(self.tag_values_dict[tag])
                elif self.tag_func_dict[tag] == "min":
                    avg_value = min(self.tag_values_dict[tag])
                elif self.tag_func_dict[tag] == "sd":
                    avg_value = np.array(self.tag_values_dict[tag]).std()

                self.summary_writer.add_scalar(tag, avg_value, global_step=self.tag_step_dict[tag])

            self.tag_step_dict[tag] += 1
            self.tag_values_dict[tag] = []

class BasicLogger:
    def __init__(self,configs) -> None:
        self.configs = configs
        self.log_dir = configs['log_dir']
        self.env_id = configs['env_id']
        self.name = configs['name']
        self.use_wandb = configs['use_wandb']
        create_path(os.path.join(self.log_dir,self.name, "summary_log"))
        self.summary_log_path = os.path.join(self.log_dir,self.name, "summary_log")

        self.summary_logger = SummaryLogger(configs, self.summary_log_path)
    
    def log_detail(self,data:dict):
        for key,value in data.items():
            self.summary_logger.add_info(key,value)
    
    def dump_summary(self):
        self.summary_logger.dump_summary()
        
        