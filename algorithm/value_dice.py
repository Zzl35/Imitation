import torch
from torch.nn.utils import clip_grad_value_
from torch.autograd import grad as torch_grad

from net import *
from algorithm.base import BaseAlgorithm


class ValueDICE(BaseAlgorithm):
    def __init__(self,
                 state_dim,
                 action_dim,
                 critic_lr=1e-3,
                 actor_lr=1e-3,
                 discount=0.99,
                 device=torch.device('cuda')):
        super().__init__(state_dim, action_dim, actor_lr, device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.update_epoch = 0

        self.discount = discount

    def _gradient_penalty(self, expert_inputs, expert_next_inputs, policy_inputs, policy_next_inputs, LAMBDA=10):
        batch_size = expert_inputs.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1).requires_grad_()
        alpha = alpha.expand_as(expert_inputs).to(self.device)
        critic_inter = alpha * expert_inputs + (1 - alpha) * policy_inputs
        critic_next_inter = alpha * expert_next_inputs + (1 - alpha) * policy_next_inputs

        interpolated = torch.cat([critic_inter, critic_next_inter], 0)

        # Calculate probability of interpolated examples
        prob_interpolated = self.critic.Q1(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Return gradient penalty
        return LAMBDA * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def update(self,
               expert_buffer,
               policy_buffer,
               replay_buffer,
               logger=None,
               alpha=0.1,
               tau=0.005
               ) -> None:
        expert_state, expert_action, expert_next_state, expert_mask, _ = expert_buffer.sample()
        policy_state, policy_action, policy_next_state, policy_mask, _ = policy_buffer.sample()
        init_state = expert_state

        expert_mask = torch.maximum(torch.zeros_like(expert_mask), expert_mask)
        policy_mask = torch.maximum(torch.zeros_like(policy_mask), policy_mask)

        init_action = self.actor.act(init_state)[1]
        expert_next_action = self.actor.act(expert_next_state)[1] * expert_mask
        policy_next_action = self.actor.act(policy_next_state)[1] * policy_mask
 
        expert_state_action = torch.cat([expert_state, expert_action], 1)
        expert_next_state_action = torch.cat([expert_next_state, expert_next_action], 1)
        policy_state_action = torch.cat([policy_state, policy_action], 1)
        policy_next_state_action = torch.cat([policy_next_state, policy_next_action], 1)

        init_value = self.critic.Q1(torch.cat([init_state, init_action], 1))

        expert_value = self.critic.Q1(expert_state_action)
        expert_next_value = self.critic.Q1(expert_next_state_action)
        expert_diff = expert_value - self.discount * expert_next_value

        policy_value = self.critic.Q1(policy_state_action)
        policy_next_value = self.critic.Q1(policy_next_state_action)
        policy_diff = policy_value - self.discount * policy_next_value

        non_linear_loss = torch.log(torch.mean((1 - alpha) * torch.exp(expert_diff) + alpha * torch.exp(policy_diff)))
        linear_loss = torch.mean((1 - alpha) * (1 - self.discount) * init_value + alpha * policy_diff)

        loss = non_linear_loss - linear_loss

        gradient_penalty = self._gradient_penalty(expert_state_action.detach(), expert_next_state_action.detach(),
                                                  policy_state_action.detach(), policy_next_state_action.detach())

        reg = orthogonal_regularization(self.actor, reg_coef=1e-4, device=self.device)

        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        loss.backward()
        clip_grad_value_(self.critic.parameters(), 40)
        for param in self.actor.parameters():
            param.grad = -param.grad
        gradient_penalty.backward()
        reg.backward()

        self.critic_optimizer.step()
        self.actor_optimizer.step()

        self.update_epoch += 1

        if self.update_epoch % 100 == 0:
            logger.add_scalar("loss/linear loss", linear_loss.item(), self.update_epoch)
            logger.add_scalar("loss/non linear loss", non_linear_loss.item(), self.update_epoch)
            logger.add_scalar("loss/critic loss", loss.item(), self.update_epoch)
            logger.add_scalar("loss/actor loss", -loss.item(), self.update_epoch)

            logger.add_scalar("watch/gradient penalty", gradient_penalty.item(), self.update_epoch)
            logger.add_scalar("watch/orthogonal regularization", reg.item(), self.update_epoch)

