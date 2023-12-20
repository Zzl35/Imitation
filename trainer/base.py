from abc import abstractmethod
import torch


class Trainer(object):
    def __init__(self, logger):
        self.logger = logger

    def reset_trainer(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                self.__setattr__(key, value)

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass


class RLTrainer(Trainer):
    def __init__(self,
                 train_env,
                 eval_env,
                 expert_buffer,
                 policy_buffer,
                 logger):
        super().__init__(logger)

        self.train_env = train_env
        self.eval_env = eval_env

        self.expert_buffer = expert_buffer
        self.policy_buffer = policy_buffer

    @abstractmethod
    def train(self):
        pass

    def evaluate(self, actor=None, eval_num=10, device=torch.device('cuda')):
        eval_length, eval_reward = 0, 0.
        for _ in range(eval_num):
            state = self.eval_env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    s = torch.FloatTensor(state.reshape(1, -1)).to(device)
                    action = actor.act(s)[0].cpu().data.numpy().flatten()
                state, reward, done, _ = self.eval_env.step(action)
                eval_reward += reward
                eval_length += 1
        return eval_length / eval_num, eval_reward / eval_num



