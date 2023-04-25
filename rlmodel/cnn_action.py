import gym
import numpy as np
import torch
import torch.nn as nn

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CNN_Encoder(nn.Module):
    def __init__(self,input_shape,output_dim):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
        )
        ## input_shape = (3,240,256)
        self.input_shape = input_shape
        self.in_dim = self._get_conv_output(self.input_shape)
        self.fc = layer_init(nn.Linear(self.in_dim, output_dim))

    def _get_conv_output(self, shape):
        # Calculate the size of the feature maps output by the convolutional layers
        batch_size = 1
        input_data = torch.rand(batch_size, *shape)
        output_feat = self.network(input_data)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def forward(self,x):
        #! x.shape = (bz,C,H,W)
        x = self.network(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    