import argparse
import json 
import os 
from distutils.util import strtobool


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='MAS',
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=bool, default=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--device", type=str,default='cuda:3')
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="my-test-project",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default='jingtianji',
        help="the entity (team) of wandb's project")
    parser.add_argument("--use_mas", type=lambda x: bool(strtobool(x)), default=True,)
    parser.add_argument("--allow_retrain", type=lambda x: bool(strtobool(x)), default=False,)

    # Algorithm specific arguments
    parser.add_argument("--total_timesteps", type=int, default=5000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num_envs", type=int, default=32,
        help="the number of parallel game environments")
    parser.add_argument("--num_steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=bool, default=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--minibatch_size", type=int, default=512,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=8,
        help="the K epochs to update the policy")
    parser.add_argument('--pi_update_frq',type = int ,default=2)
    parser.add_argument("--norm-adv", type=bool, default=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=bool, default=False,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.02,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    # fmt: on
    return args


class Parameters(object):
    def __init__(self,env_id,config_path = None,save_path = "logs"):
        #! parameter priotity: config file > command line
        args = parse_args()
        args = args._get_kwargs()
        default_config = dict()
        for arg in args:
            default_config[arg[0]] = arg[1]
        if config_path is not None and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            for k,newvalue in loaded_config.items():
                default_config[k] = newvalue
        self.args = default_config
        self.args['env_id'] = env_id
        self.short_name = self.args['env_id'] + '_' +self.args.get('exp_name','Test')
        self.json_name = 'parameter.json'

        self.apply_vars(self.args)
        self.save_path =  os.path.join(save_path,self.short_name)
        if save_path is not None:
            self.save_config(self.save_path)
        self.model_path = self.save_path

    def save_config(self,path):
        
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, self.json_name), 'w') as f:
            ser = json.dumps(self.args)
            f.write(ser)
    def load_config(self,path):
        assert os.path.exists(path)
        with open(os.path.join(path, self.json_name), 'r') as f:
            ser = json.load(f)
            self.args = ser

    def apply_vars(self, d):
        for k,v in d.items():
            setattr(self, k, v)