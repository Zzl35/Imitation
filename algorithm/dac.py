import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
from torch.autograd import grad as torch_grad

from net import *
from net.utils import orthogonal_regularization
from algorithm.base import BaseAlgorithm


class DAC(BaseAlgorithm):
    def __init__(self,
                 state_dim,
                 action_dim,
                 discriminator_lr=1e-3,
                 critic_lr=1e-3,
                 actor_lr=1e-3,
                 alpha_lr=1e-3,
                 discount=0.99,
                 ensemble_num=2,
                 lamda=10,
                 init_temp=1,
                 learn_temp=False,
                 device=torch.device('cuda')):
        super().__init__(state_dim, action_dim, device)

        self.discriminator = Discriminator(state_dim, action_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=discriminator_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.actor = StochasticActor(state_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=1e-3)
      
        self.log_alpha = torch.tensor(np.log(init_temp)).to(self.device)
        self.log_alpha.requires_grad = True
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        
        self.learn_temp = learn_temp
        self.lamda = lamda

        self.target_entropy = -self.action_dim
        self.update_step = 0
        self.pretrain_step = 0

        self.discount = discount

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
    
    def getV(self, state):
        _, action, log_prob = self.actor.act(state)
        q = torch.min(*self.critic(torch.cat([state, action], dim=-1)))
        v = q - self.alpha.detach() * log_prob
        return v
    
    def cql_loss(self, state, action):
        q = torch.min(*self.critic(torch.cat([state, action], dim=-1)))
        v = self.getV(state)
        return (v - q).mean()

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

        c_mask = torch.maximum(torch.zeros_like(mask), mask)
        a_mask = 1.0 - torch.maximum(torch.zeros_like(mask), -mask)

        with torch.no_grad():
            reward = self.reward(state, action, next_state)
            # Select action according to policy and add clipped noise
            _, next_action, log_prob = self.actor.act(next_state)

            # Compute the target Q value
            # next_state_action = torch.cat([next_state, next_action * c_mask], 1)
            # next_q1, next_q2 = self.critic_target(next_state_action)
            q1, q2 = self.critic_target(torch.cat([next_state, next_action * c_mask], 1))
            next_q = torch.min(q1, q2) - self.alpha * log_prob
            target_q = (reward + self.discount * next_q)

        loss_dict['watch/discriminator reward'] = reward.mean().item()

        # Compute the target Q value
        current_q1, current_q2 = self.critic(torch.cat([state, action], 1))
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        cql_loss = self.cql_loss(expert_state, expert_action)
        loss = cql_loss + critic_loss
        
        with torch.no_grad():
            expert_q = torch.min(*self.critic(torch.cat([expert_state, expert_action], dim=-1)))
            sample_action = self.actor.act(expert_state)[1]
            sample_expert_q = torch.min(*self.critic(torch.cat([expert_state, sample_action], dim=-1)))
            random_action = expert_action.clone().uniform_(-1, 1)
            random_expert_q = torch.min(*self.critic(torch.cat([expert_state, random_action], dim=-1)))
        loss_dict["watch/expert q"] = expert_q.mean()
        loss_dict["watch/expert sample q"] = sample_expert_q.mean()
        loss_dict["watch/expert random q"] = random_expert_q.mean()
        loss_dict["watch/current q"] = current_q1.mean()
        loss_dict["watch/target q"] = target_q.mean()
        loss_dict["loss/critic loss"] = critic_loss.item()
        loss_dict["loss/cql loss"] = cql_loss.item()
        # loss = critic_loss
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        loss.backward()
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

        return loss_dict

    def bc_update(self, expert_buffer):
        expert_state, expert_action, _, _, _ = expert_buffer.get_samples()
        loss = -self.actor.log_prob(expert_state, expert_action).mean()

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

