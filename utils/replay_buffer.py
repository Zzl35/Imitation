from collections import deque
import numpy as np
import random
import torch

from utils.expert_dataset import ExpertDataset


class Memory(object):
    def __init__(self, memory_size: int, seed: int = 0, device=torch.device('cpu')) -> None:
        random.seed(seed)
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)
        self.device = device

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int = 42, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

    def save(self, path):
        b = np.asarray(self.buffer)
        print(b.shape)
        np.save(path, b)

    def load(self, path, num_trajs, sample_freq, seed):
        # If path has no extension add npy
        if not path.endswith("pkl"):
            path += '.npy'
        data = ExpertDataset(path, num_trajs, sample_freq, seed)
        # data = np.load(path, allow_pickle=True)
        for i in range(len(data)):
            self.add(data[i])

    def get_samples(self, batch_size=256):
        batch = self.sample(batch_size, False)

        batch_state, batch_action, batch_next_state, batch_mask, batch_reward = zip(
            *batch)

        # Scale obs for atari. TODO: Use flags
        batch_state = np.array(batch_state)
        batch_next_state = np.array(batch_next_state)
        batch_action = np.array(batch_action)

        batch_state = torch.as_tensor(batch_state, dtype=torch.float, device=self.device)
        batch_next_state = torch.as_tensor(batch_next_state, dtype=torch.float, device=self.device)
        batch_action = torch.as_tensor(batch_action, dtype=torch.float, device=self.device)
        if batch_action.ndim == 1:
            batch_action = batch_action.unsqueeze(1)
        batch_reward = torch.as_tensor(batch_reward, dtype=torch.float, device=self.device).unsqueeze(1)
        batch_mask = torch.as_tensor(batch_mask, dtype=torch.float, device=self.device).unsqueeze(1)

        return batch_state, batch_action, batch_next_state, batch_mask, batch_reward


# class ReplayBuffer(object):
#     def __init__(self,
#                  state_dim,
#                  action_dim,
#                  max_size=int(1e6),
#                  device=torch.device('cuda')):
#         self.state_dim = state_dim
#         self.action_dim = action_dim

#         self.max_size = max_size
#         self.ptr = 0
#         self.size = 0

#         self.state = np.zeros((max_size, state_dim))
#         self.action = np.zeros((max_size, action_dim))
#         self.next_state = np.zeros((max_size, state_dim))
#         self.mask = np.zeros((max_size, 1))
#         self.weight = np.zeros((max_size, 1))

#         self.device = device

#     def __len__(self):
#         return self.size

#     def clear(self):
#         self.size = 0
#         self.ptr = 0

#     def reset_weight(self, weight):
#         if hasattr(weight, '__len__'):
#             assert len(weight) == self.size
#             weight = np.array(weight).reshape(self.size, 1)
#         self.weight[: self.size] = weight

#     def add(self, state, action, next_state, mask, weight=1.):
#         self.state[self.ptr] = state
#         self.action[self.ptr] = action
#         self.next_state[self.ptr] = next_state
#         self.mask[self.ptr] = mask
#         self.weight[self.ptr] = weight

#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)

#     def sample(self, batch_size=256):
#         ind = np.random.randint(0, self.size, size=batch_size)

#         return (
#             torch.FloatTensor(self.state[ind]).to(self.device),
#             torch.FloatTensor(self.action[ind]).to(self.device),
#             torch.FloatTensor(self.next_state[ind]).to(self.device),
#             torch.FloatTensor(self.mask[ind]).to(self.device),
#             torch.FloatTensor(self.weight[ind]).to(self.device)
#         )

#     def get_all(self):
#         ind = [i for i in range(self.size)]

#         return (
#             torch.FloatTensor(self.state[ind]).to(self.device),
#             torch.FloatTensor(self.action[ind]).to(self.device),
#             torch.FloatTensor(self.next_state[ind]).to(self.device),
#             torch.FloatTensor(self.mask[ind]).to(self.device),
#             torch.FloatTensor(self.weight[ind]).to(self.device)
#         )






