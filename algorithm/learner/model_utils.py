import torch as th 
import torch.nn as nn
import os 
import io 
import numpy as np  

def to_device(data, device="cpu"):
    """全部数据x（numpy.ndarray）调用torch.from_numpy(x).to(device)"""
    if isinstance(data, dict):
        result = dict()
        for key, value in data.items():
            result[key] = to_device(value, device)
    elif isinstance(data, (list, tuple)):
        result = [to_device(d, device) for d in data]
    else:
        result = th.from_numpy(data).float().to(device)
    return result

def load_script_model(path):
    with open(path, 'rb') as m:
        models = io.BytesIO(m.read())
    net = th.jit.load(models, map_location=th.device('cpu'))
    return net

def save_script_model(path,model):
    script_net = th.jit.script(model)
    th.jit.save(script_net, path)

def serialize_model(net, path_prefix, name):
    # timestamp = time.strftime("%y%m%d%H%M%S", time.localtime(time.time()))
    # file_name = saved_path_prefix + '_' + timestamp
    if os.path.exists(path_prefix) is False:
        os.makedirs(path_prefix)
    path = os.path.join(path_prefix, name)
    save_script_model(path, net)

def deserialize_model(path_prefix, name):
    path = os.path.join(path_prefix, name)
    net = load_script_model(path)
    return net