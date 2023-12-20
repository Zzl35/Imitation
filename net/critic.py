import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
            

class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


class EnsembleCritic(nn.Module):
    def __init__(self, state_dim, action_dim, ensemble_num=2, hidden_dim=256, hidden_layer=2):
        super(EnsembleCritic, self).__init__()
        
        self.ensemble_num = ensemble_num
        
        self.ensemble_q = nn.ModuleList()
        for _ in range(ensemble_num):
            self.ensemble_q.append(mlp(state_dim+action_dim, hidden_dim, 1, hidden_layer))
    
    def forward(self, x):
        q = []
        for qnet in self.ensemble_q:
            q.append(qnet(x))
        return torch.stack(q)
        
        
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
        
        self._init_net()
        
    def _init_net(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                nn.init.orthogonal_(layer.weight)

    def forward(self, x):
        q1 = F.relu(self.l1(x))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(x))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, x):
        q1 = F.relu(self.l1(x))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1