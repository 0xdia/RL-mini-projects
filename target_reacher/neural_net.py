import torch
import numpy as np

class QNeuralNetwork(torch.nn.Module):
    def __init__(self, state_size, action_size, hidden_leyers_sizes, seed):
        super(QNeuralNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(state_size+action_size, hidden_leyers_sizes[0]))
        for i in range(1, len(hidden_leyers_sizes)):
            self.layers.append(torch.nn.Linear(hidden_leyers_sizes[i-1], hidden_leyers_sizes[i]))
        self.layers.append(torch.nn.Linear(hidden_leyers_sizes[-1], 1))
    
    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        for l in range(len(self.layers)-1):
            x = torch.nn.functional.relu(self.layers[l](x))
        return self.layers[-1](x)