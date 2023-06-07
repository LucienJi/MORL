import os 
import json 
import wandb
from algorithm.logger.utils import setup_logger,create_path
import tensorflow as tf
import time 
class SummaryLogger:
    def __init__(self,configs,summary_file_to_write) -> None:
        self.configs = configs
        self.summary_writer = tf.summary.create_file_writer(summary_file_to_write)

class BasicLogger:
    def __init__(self,configs) -> None:
        self.configs = configs
        self.log_dir = configs['log_dir']
        self.env_id = configs['env_id']
        self.name = configs['name']
        self.use_wandb = configs['use_wandb']
        create_path(os.path.join(self.log_dir, "summary_log"))
        self.summary_log_path = os.path.join(self.log_dir, "summary_log")
        
        self.tag_values_dict = {}
        self.tag_step_dict = {}
        self.tag_output_threshold_dict = {}
        self.tag_func_dict = {}
        self.tag_total_add_count = {}
        self.total_tag_type = [
            "time_avg", "time_total",
            "avg", "total", "max", "min"
        ]
    def add_tag(self, tag, output_threshold = 0, cal_type = 'avg', time_threshold=0, bins=100):
        self.tag_values_dict[tag] = []
        self.tag_step_dict[tag] = 0
        self.tag_output_threshold_dict[tag] = output_threshold
        self.tag_func_dict[tag] = cal_type
        self.tag_total_add_count[tag] = 0
    
    def add_summary(self, tag, value, timestamp=time.time()):
        self.tag_values_dict[tag].append(value)
        self.tag_total_add_count[tag] += 1
        if self.tag_func_dict[tag].startswith("time"):
            self.tag_time_data_timestamp[tag].append(timestamp)

        if self.tag_func_dict[tag].startswith("time") is False and \
                len(self.tag_values_dict[tag]) >= self.tag_output_threshold_dict[tag]:
            if self.tag_func_dict[tag].find("histogram") != -1:
                # each value is a list
                all_values = []
                for i in self.tag_values_dict[tag]:
                    all_values.extend(i)
                with self.summary_writer.as_default():
                    tf.summary.histogram(tag, all_values, step=self.tag_step_dict[tag])
                # self.log_histogram(tag, all_values, self.tag_step_dict[tag], self.tag_bins[tag])
            else:
                summary = tf.compat.v1.Summary()
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

                with self.summary_writer.as_default():
                    tf.summary.scalar(tag, avg_value, step=self.tag_step_dict[tag])

            self.tag_step_dict[tag] += 1
            self.tag_values_dict[tag] = []

        