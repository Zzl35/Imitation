import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()

        self.linear1 = nn.Linear(state_dim + action_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)
        
        self._init_net()
        
    def _init_net(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                nn.init.orthogonal_(layer.weight)

    def forward(self, x):
        x = torch.tanh(self.linear1(x).clip(-10, 10))
        x = torch.tanh(self.linear2(x).clip(-10, 10))
        out = self.linear3(x)
        return out

    def reward(self, x):
        return self.forward(x)
        # return torch.log(probs + 1e-8) - torch.log(1 - probs + 1e-8)
        # return self(x)

