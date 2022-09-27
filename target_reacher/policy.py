import torch


class Policy(torch.nn.Module):
    def __init__(self, state_size, action_size, hidden_layers_sizes, seed):
        super(Policy, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(state_size, hidden_layers_sizes[0]))
        for i in range(1, len(hidden_layers_sizes)):
            self.layers.append(torch.nn.Linear(hidden_layers_sizes[i-1], hidden_layers_sizes[i]))
        self.layers.append(torch.nn.Linear(hidden_layers_sizes[-1], action_size))
    
    def forward(self, state):
        x = state
        for l in range(len(self.layers)-1):
            x = torch.nn.functional.relu(self.layers[l](x))
        return self.layers[-1](x)
