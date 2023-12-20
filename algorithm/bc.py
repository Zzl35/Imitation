import torch
import torch.nn.functional as F

from net.actor import DeterministicActor, StochasticActor
from net.utils import orthogonal_regularization
from algorithm.base import BaseAlgorithm


class BC(BaseAlgorithm):
    def __init__(self, state_dim, action_dim, actor_lr=1e-3, device=torch.device('cuda')):
        super().__init__(state_dim, action_dim, device)
        
        # self.actor = DeterministicActor(state_dim, action_dim).to(device)
        self.actor = StochasticActor(state_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=1e-3)

    def update(self,
               expert_buffer=None):
        expert_state, expert_action, _, _, _ = expert_buffer.get_samples()
        loss = -torch.mean(self.actor.log_prob(expert_state, expert_action))
        # loss = F.mse_loss(self.actor(expert_state), expert_action)

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {"loss/bc loss": loss.item()}
