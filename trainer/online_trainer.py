from tqdm import tqdm
import numpy as np
import torch

from trainer.base import RLTrainer
from algorithm import *


class OnlineRLTrainer(RLTrainer):
    def __init__(self, train_env, eval_env, expert_buffer, policy_buffer, replay_buffer, logger):
        super().__init__(train_env, eval_env, expert_buffer, policy_buffer, logger)

        self.replay_buffer = replay_buffer

    def train(self,
              algo=None,
              num_epoch=int(2e5),
              random_epoch=1000,
              start_epoch=2000,
              absorbing_per_episode=10,
              update_per_step=1,
              log_interval=100,
              save_model=False):
        with tqdm(total=num_epoch) as pbar:
            pbar.set_description("Start Online Training!")

            training_length, training_reward = [], []
            episode_length, episode_reward = 0, 0
            best_returns = -9999.

            state = self.train_env.reset()
            for epoch in range(num_epoch):
                if epoch < random_epoch:
                    action = self.train_env.action_space.sample()
                else:
                    s = torch.FloatTensor(state.reshape(1, -1)).to(algo.device)
                    action = algo.actor.act(s)[1].cpu().data.numpy().flatten()
                next_state, reward, done, _ = self.train_env.step(action)

                episode_length += 1
                episode_reward += reward

                done_bool = float(done) if self.train_env.episode < self.train_env.max_episode else 0
                if done_bool and hasattr(self.train_env, 'absorbing_state'):
                    next_state = self.train_env.absorbing_state

                self.policy_buffer.add((state, action, next_state, 1. - done_bool, 1.))
                self.replay_buffer.add((state, action, next_state, 1. - done_bool, 1.))

                state = next_state

                if done_bool:
                    for _ in range(min(self.train_env.max_episode - self.train_env.episode, absorbing_per_episode)):
                        absorbing_state = self.train_env.absorbing_state
                        absorbing_action = self.train_env.absorbing_action
                        self.policy_buffer.add((absorbing_state, absorbing_action, absorbing_state, -1, 1.))
                        self.replay_buffer.add((absorbing_state, absorbing_action, absorbing_state, -1, 1.))

                if done:
                    state = self.train_env.reset()

                    training_length.append(episode_length)
                    training_reward.append(episode_reward)

                    episode_length, episode_reward = 0, 0

                if epoch > start_epoch:
                    for _ in range(update_per_step):
                        loss_dict = algo.update(self.expert_buffer, self.policy_buffer, self.replay_buffer)
                            
                    if not epoch % 100:
                        for k, v in loss_dict.items():
                            self.logger.add_scalar(k, v, epoch)

                # evaluate policy
                if not epoch % log_interval:
                    eval_length, eval_returns = self.evaluate(algo.actor, device=algo.device)
                    self.logger.logkv('eval env/finetune returns', eval_returns)
                    pbar.set_description(("eval returns: {:.4f}".format(eval_returns)))

                    if eval_returns > best_returns and save_model:
                        algo.save(self.logger._model_dir)
                        best_returns = eval_returns

                    if len(training_length) > 0:
                        avg_num = min(len(training_length), 5)
                        self.logger.logkv('train env/finetune returns', np.mean(training_reward[-avg_num:]))

                    self.logger.set_timestep(epoch)
                    self.logger.dumpkvs(exclude=["pretraining_progress"])

                pbar.update(1)
