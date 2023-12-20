import torch
from torch.nn.utils import clip_grad_value_
from torch.autograd import grad as torch_grad

from net import *
from algorithm.base import BaseAlgorithm


class IQLearn(BaseAlgorithm):
    def __init__(self,
                 state_dim,
                 action_dim,
                 critic_lr=1e-3,
                 actor_lr=1e-3,
                 alpha_lr=1e-3,
                 discount=0.99,
                 device=torch.device('cuda')):
        super().__init__(state_dim, action_dim, actor_lr, device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.log_alpha = torch.zeros((1,), requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.target_entropy = -self.action_dim
        self.update_step = 0

        self.discount = discount

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    def get_v(self, state, mask=None, use_target=False):
        _, action, log_prob = self.actor.act(state)
        action = mask * action if mask is not None else action
        state_action = torch.cat([state, action], 1).to(self.device)
        if use_target:
            q1, q2 = self.critic_target(state_action)
        else:
            q1, q2 = self.critic(state_action)
        v = torch.min(q1, q2) - self.alpha.detach() * log_prob
        return v

    def _gradient_penalty(self, expert_inputs, policy_inputs, LAMBDA=5):
        batch_size = expert_inputs.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1).requires_grad_()
        alpha = alpha.expand_as(expert_inputs).to(self.device)
        interpolated = alpha * expert_inputs + (1 - alpha) * policy_inputs

        # Calculate probability of interpolated examples
        prob_interpolated1, prob_interpolated2 = self.critic(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated1, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated1.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0] + \
                    torch_grad(outputs=prob_interpolated2, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated2.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]
        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Return gradient penalty
        return LAMBDA * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def critic_update(self, expert_buffer, policy_buffer, tau=0.005):
        loss_dict = {}

        expert_state, expert_action, expert_next_state, expert_mask, _ = expert_buffer.sample()
        policy_state, policy_action, policy_next_state, policy_mask, _ = policy_buffer.sample()

        # expert_a_mask = torch.maximum(torch.zeros_like(expert_mask), expert_mask)
        # expert_c_mask = 1. - torch.maximum(torch.zeros_like(expert_mask), -expert_mask)
        #
        # policy_a_mask = torch.maximum(torch.zeros_like(policy_mask), policy_mask)
        # policy_c_mask = 1. - torch.maximum(torch.zeros_like(policy_mask), -policy_mask)

        ######
        # IQ-Learn minimal implementation with X^2 divergence (~15 lines)
        # Calculate 1st term of loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
        expert_current_q1, expert_current_q2 = self.critic(torch.cat([expert_state, expert_action], 1))
        expert_next_v = self.get_v(expert_next_state, True)

        reward1 = (expert_current_q1 - self.discount * expert_next_v)
        reward2 = (expert_current_q2 - self.discount * expert_next_v)
        loss = -torch.mean(reward1 + reward2)

        # 2nd term for our loss (use expert and policy states): E_(ρ)[Q(s,a) - γV(s')]
        expert_current_v = self.get_v(expert_state)
        policy_current_v = self.get_v(policy_state)
        policy_next_v = self.get_v(policy_next_state, True)
        value_loss = torch.mean(expert_current_v - expert_mask * self.discount * expert_next_v) + \
                     torch.mean(policy_current_v - policy_mask * self.discount * policy_next_v)
        loss += value_loss

        # Use χ2 divergence (adds a extra term to the loss)
        chi2_loss = 0.5 * (reward1 ** 2 + reward2 ** 2).mean()
        loss += chi2_loss
        ######

        gradient_penalty = self._gradient_penalty(torch.cat([expert_state, expert_action], 1),
                                                  torch.cat([policy_state, policy_action], 1))
        loss += gradient_penalty

        loss_dict['reward'] = torch.mean(reward1 + reward2).item()
        loss_dict['value loss'] = value_loss.item()
        loss_dict['chi2 loss'] = chi2_loss.item()
        loss_dict['gradient penalty'] = gradient_penalty.item()
        loss_dict['critic loss'] = loss.item()

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        soft_update(self.critic, self.critic_target, tau)

        return loss_dict

    # def actor_update(self, expert_batch, policy_batch, actor_grad_clipping=40):
    #     loss_dict = {}

    #     expert_state, _, _, expert_mask, _ = expert_batch
    #     policy_state, _, _, policy_mask, _ = policy_batch

    #     state = torch.cat([expert_state, policy_state], 0)
    #     mask = torch.cat([expert_mask, policy_mask], 0)

    #     mask = 1.0 - torch.maximum(torch.zeros_like(mask), -mask)

    #     _, action, log_prob = self.actor.act(state)
    #     q1, q2 = self.critic(torch.cat([state, action], 1))
    #     q = torch.min(q1, q2) - self.alpha * log_prob
    #     actor_loss = -torch.sum(mask * q) / torch.sum(mask)
    #     reg = orthogonal_regularization(self.actor, device=self.device)
    #     loss_dict['actor loss'] = actor_loss.item()
    #     loss_dict['orthogonal regularization'] = reg.item()
    #     actor_loss += reg

    #     # Optimize the actor
    #     self.actor_optimizer.zero_grad()
    #     actor_loss.backward()
    #     clip_grad_value_(self.actor.parameters(), actor_grad_clipping)
    #     self.actor_optimizer.step()

    #     # optimize alpha
    #     alpha_loss = -torch.sum(mask * self.alpha * (log_prob.detach() + self.target_entropy)) / torch.sum(mask)
    #     self.alpha_optimizer.zero_grad()
    #     alpha_loss.backward()
    #     self.alpha_optimizer.step()
    #     loss_dict['alpha loss'] = alpha_loss.item()

    #     return loss_dict

    def actor_update(self, replay_buffer, actor_grad_clipping=40):
        loss_dict = {}

        state, _, _, mask, _ = replay_buffer.sample()

        mask = 1.0 - torch.maximum(torch.zeros_like(mask), -mask)

        _, action, log_prob = self.actor.act(state)
        q1, q2 = self.critic(torch.cat([state, action], 1))
        q = torch.min(q1, q2) - self.alpha * log_prob
        actor_loss = -torch.sum(mask * q) / torch.sum(mask)
        reg = orthogonal_regularization(self.actor, device=self.device)
        loss_dict['actor loss'] = actor_loss.item()
        loss_dict['orthogonal regularization'] = reg.item()
        actor_loss += reg

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_value_(self.actor.parameters(), actor_grad_clipping)
        self.actor_optimizer.step()

        # optimize alpha
        alpha_loss = -torch.sum(mask * self.alpha * (log_prob.detach() + self.target_entropy)) / torch.sum(mask)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        loss_dict['alpha loss'] = alpha_loss.item()

        return loss_dict

    def update(self,
               expert_buffer,
               policy_buffer,
               replay_buffer,
               logger=None
               ) -> None:

        expert_batch = expert_buffer.sample()
        policy_batch = policy_buffer.sample()

        critic_loss_dict = self.critic_update(expert_buffer, policy_buffer)
        actor_loss_dict = self.actor_update(replay_buffer)

        self.update_step += 1

        logger.add_scalar("loss/value loss", critic_loss_dict['value loss'], self.update_step)
        logger.add_scalar("loss/reward", critic_loss_dict['reward'], self.update_step)
        logger.add_scalar("loss/chi2 loss", critic_loss_dict['chi2 loss'], self.update_step)
        logger.add_scalar("loss/critic loss", critic_loss_dict['critic loss'], self.update_step)
        logger.add_scalar("loss/actor loss", actor_loss_dict['actor loss'], self.update_step)
        logger.add_scalar("loss/alpha loss", actor_loss_dict['alpha loss'], self.update_step)

        logger.add_scalar("watch/gradient penalty", critic_loss_dict['gradient penalty'], self.update_step)
        logger.add_scalar("watch/orthogonal regularization", actor_loss_dict['orthogonal regularization'], self.update_step)
        logger.add_scalar("watch/alpha", self.alpha.item(), self.update_step)

