import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
from torch.autograd import grad as torch_grad

from net import *
from algorithm.base import BaseAlgorithm


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


class DAC(BaseAlgorithm):
    def __init__(self,
                 state_dim,
                 action_dim,
                 discriminator_lr=1e-3,
                 critic_lr=1e-3,
                 actor_lr=1e-3,
                 alpha_lr=1e-3,
                 discount=0.99,
                 lamda=10,
                 init_temp=1,
                 learn_temp=True,
                 use_cql=True,
                 n_actions=10,
                 device=torch.device('cuda')):
        super().__init__(state_dim, action_dim, device)

        self.discriminator = Discriminator(state_dim, action_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=discriminator_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.actor = StochasticActor(state_dim, action_dim).to(device)
        self.actor_target = StochasticActor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
      
        self.log_alpha = torch.tensor(np.log(init_temp)).to(self.device)
        self.log_alpha.requires_grad = True
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        
        self.learn_temp = learn_temp
        self.lamda = lamda

        self.target_entropy = -self.action_dim
        self.update_step = 0

        self.discount = discount
        
        self.use_cql = use_cql
        self.n_actions = n_actions

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    def _gradient_penalty(self, real_data, generated_data, LAMBDA=10):
        batch_size = real_data.size()[0]

        # Calculate interpolationsubsampling_rate=20
        alpha = torch.rand(batch_size, 1).requires_grad_()
        alpha = alpha.expand_as(real_data).to(self.device)
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data

        # Calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Return gradient penalty
        return LAMBDA * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def reward(self, state, action, next_state):
        with torch.no_grad():
            state_action = torch.cat([state, action], 1).to(self.device)
            reward = self.discriminator.reward(state_action)
        return reward

    def cql_loss(self, state, action, next_state):
        batch_size = action.shape[0]
        action_dim = action.shape[-1]
        cql_random_actions = action.new_empty(
            (batch_size, self.n_actions, action_dim), requires_grad=False
        ).uniform_(-1, 1)
        _, cql_current_actions, cql_current_log_pis = self.actor.act(
            state, n_actions=self.n_actions
        )
        _, cql_next_actions, cql_next_log_pis = self.actor.act(
            next_state, n_actions=self.n_actions
        )
        cql_current_actions, cql_current_log_pis = (
            cql_current_actions.detach(),
            cql_current_log_pis.detach(),
        )
        cql_next_actions, cql_next_log_pis = (
            cql_next_actions.detach(),
            cql_next_log_pis.detach(),
        )
        log_probs = self.actor.log_prob(state, action)

        q1_predicted, q2_predicted = self.critic(torch.cat([state, action], dim=-1))
        cql_q1_rand, cql_q2_rand = self.critic(torch.cat([extend_and_repeat(state, 1, self.n_actions), cql_random_actions], dim=-1))
        cql_q1_current_actions, cql_q2_current_actions = self.critic(torch.cat([extend_and_repeat(state, 1, self.n_actions), cql_current_actions], dim=-1))
        cql_q1_next_actions, cql_q2_next_actions = self.critic(torch.cat([extend_and_repeat(state, 1, self.n_actions), cql_next_actions], dim=-1))
        
        q1 = torch.cat([cql_q1_rand, cql_q1_current_actions, cql_q1_next_actions], dim=1)
        q2 = torch.cat([cql_q2_rand, cql_q2_current_actions, cql_q2_next_actions], dim=1)
        q1_max = torch.max(q1, dim=1)[0]
        q2_max = torch.max(q2, dim=1)[0]

        random_density = np.log(0.5**action_dim)
        cql_cat_q1 = torch.cat(
            [
                cql_q1_rand - random_density * self.alpha,
                q1_predicted.unsqueeze(dim=1) - log_probs.detach() * self.alpha,
                cql_q1_next_actions - cql_next_log_pis.detach() * self.alpha,
                cql_q1_current_actions - cql_current_log_pis.detach() * self.alpha,
            ],
            dim=1,
        )
        cql_cat_q2 = torch.cat(
            [
                cql_q2_rand - random_density * self.alpha,
                q2_predicted.unsqueeze(dim=1) - log_probs.detach() * self.alpha,
                cql_q2_next_actions - cql_next_log_pis.detach() * self.alpha,
                cql_q2_current_actions - cql_current_log_pis.detach() * self.alpha,
            ],
            dim=1,
        )
            
        # cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        # cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf1_ood = torch.mean(cql_cat_q1, dim=1)
        cql_qf2_ood = torch.mean(cql_cat_q2, dim=1)

        """Subtract the log likelihood of data"""
        cql_qf1_diff = cql_qf1_ood - q1_predicted
        cql_qf2_diff = cql_qf2_ood - q2_predicted

        qf_loss = cql_qf1_diff + cql_qf2_diff

        return qf_loss.mean(), q1_max.mean(), q2_max.mean(), q1_predicted.mean(), q2_predicted.mean()

    def discriminator_update(self, expert_buffer, policy_buffer):
        loss_dict = {}
        
        # Sample replay buffer
        policy_state, policy_action, _, _, _ = policy_buffer.get_samples()

        # Sample expert buffer
        expert_state, expert_action, _, _, _ = expert_buffer.get_samples()

        # Predict
        policy_state_action = torch.cat([policy_state, policy_action], 1).to(self.device)
        expert_state_action = torch.cat([expert_state, expert_action], 1).to(self.device)

        fake = self.discriminator(policy_state_action)
        real = self.discriminator(expert_state_action)

        # Gradient penalty for regularization.
        gradient_penalty = self._gradient_penalty(expert_state_action, policy_state_action, LAMBDA=self.lamda)

        # The main discriminator loss
        learner_loss = -torch.mean(torch.log(1 - torch.sigmoid(fake)))
        expert_loss = -torch.mean(torch.log(torch.sigmoid(real)))
        main_loss = learner_loss + expert_loss
        
        loss_dict['loss/discriminator loss'] = main_loss.item()
        loss_dict['loss/gradient penalty'] = gradient_penalty.item()

        total_loss = main_loss + gradient_penalty
        # total_loss = main_loss

        self.discriminator_optimizer.zero_grad()
        total_loss.backward()
        self.discriminator_optimizer.step()

        return loss_dict

    def sac_update(self, expert_buffer, policy_buffer, replay_buffer, update_actor=True, tau=0.005, actor_grad_clipping=40):
        loss_dict = {}

        # Sample replay buffer
        expert_state, expert_action, expert_next_state, expert_mask, _ = expert_buffer.get_samples()
        state, action, next_state, mask, _ = replay_buffer.get_samples()
        # policy_state, policy_action, policy_next_state, policy_mask, _ = policy_buffer.get_samples()
        # state = torch.cat([expert_state, policy_state])
        # action = torch.cat([expert_action, policy_action])
        # next_state = torch.cat([expert_next_state, policy_next_state])
        # mask = torch.cat([expert_mask, policy_mask])

        c_mask = torch.maximum(torch.zeros_like(mask), mask)
        a_mask = 1.0 - torch.maximum(torch.zeros_like(mask), -mask)

        with torch.no_grad():
            reward = self.reward(state, action, next_state)
            # Select action according to policy and add clipped noise
            _, next_action, log_prob = self.actor_target.act(next_state)

            # Compute the target Q value
            next_state_action = torch.cat([next_state, next_action * c_mask], 1)
            next_q1, next_q2 = self.critic_target(next_state_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha * log_prob
            target_q = reward + self.discount * next_q

        loss_dict['watch/discriminator reward'] = reward.mean().item()

        # Compute the target Q value
        current_q1, current_q2 = self.critic(torch.cat([state, action], 1))
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # qf_loss, q1_max, q2_max, q1_predicted, q2_predicted = self.cql_loss(expert_state, expert_action, expert_next_state)
        # if self.use_cql:
        #     qf_loss = 0
        #     loss = critic_loss + qf_loss
        # else:
        #     loss = critic_loss

        # loss_dict["watch/q1 max"] = q1_max.item()
        # loss_dict["watch/q1=2 max"] = q2_max.item()
        # loss_dict["watch/q1 predicted"] = q1_predicted.item()
        # loss_dict["watch/q2 predicted"] = q2_predicted.item()
        # loss_dict["loss/cql loss"] = qf_loss.item()
        loss_dict["loss/critic loss"] = critic_loss.item()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        _, action, log_prob = self.actor.act(state)
        q1, q2 = self.critic(torch.cat([state, action], 1))
        q = torch.min(q1, q2) - self.alpha * log_prob
        actor_loss = -torch.sum(a_mask * q) / torch.sum(a_mask)
        loss_dict['loss/actor loss'] = actor_loss.item()
        loss_dict['loss/entropy'] = -log_prob.mean().item()

        # Optimize the actor
        if update_actor:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            clip_grad_value_(self.actor.parameters(), actor_grad_clipping)
            self.actor_optimizer.step()
            
        # optimize alpha
        if self.learn_temp:
            alpha_loss = -torch.sum(a_mask * self.alpha * (log_prob.detach() + self.target_entropy)) / torch.sum(a_mask)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            loss_dict['loss/alpha loss'] = alpha_loss.item()
            
        # Update the frozen target models
        soft_update(self.critic, self.critic_target, tau)
        if update_actor:
            soft_update(self.actor, self.actor_target, tau)

        return loss_dict

    def bc_update(self, expert_buffer):
        expert_state, expert_action, _, mask, _ = expert_buffer.get_samples()
        a_mask = 1.0 - torch.maximum(torch.zeros_like(mask), -mask)
        loss = -(self.actor.log_prob(expert_state, expert_action) * a_mask).sum() / a_mask.sum()
        # reg = orthogonal_regularization(self.actor)
        # loss += reg

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        
        return {"loss/bc loss": loss.item()}

    def update(self,
               expert_buffer,
               policy_buffer,
               replay_buffer,
               update_actor=True,
               ) -> None:
        loss_dict = self.discriminator_update(expert_buffer, policy_buffer)
        loss_dict.update(self.sac_update(expert_buffer, policy_buffer, replay_buffer, update_actor))

        self.update_step += 1

        return loss_dict


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_STD_MIN = -5
LOG_STD_MAX = 2
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


class DeterministicActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.):
        super(DeterministicActor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action
        
        self._init_net()
        
    def _init_net(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                nn.init.orthogonal_(layer.weight)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x

    def act(self, x):
        action = self.forward(x)
        noise = action.clone().data.normal_(0, 0.2).clamp(-0.5, 0.5)
        sample = (action + noise).clamp(-self.max_action, self.max_action)
        return action, sample


class StochasticActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.):
        super(StochasticActor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim * 2)

        self.max_action = max_action
        self._init_net()
        
    def _init_net(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                nn.init.orthogonal_(layer.weight)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.tanh(self.l3(x))

        mu, log_std = torch.split(x, split_size_or_sections=int(x.shape[-1] / 2), dim=-1)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = torch.exp(log_std)

        return mu, std

    def act(self, state, n_actions=None):
        if n_actions is not None:
            state = extend_and_repeat(state, 1, n_actions)
            
        mu, std = self.forward(state)

        dist = torch.distributions.Normal(mu, std)
        sample = dist.rsample().clamp(-self.max_action, self.max_action)
        log_prob = torch.mean(dist.log_prob(sample), dim=-1, keepdim=True)

        return mu, sample, log_prob

    def log_prob(self, state, action):
        mu, std = self.forward(state)

        dist = torch.distributions.Normal(mu, std)
        log_prob = torch.mean(dist.log_prob(action), dim=-1, keepdim=True)

        return log_prob