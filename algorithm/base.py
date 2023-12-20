import os
import torch
import torch.nn as nn
from abc import abstractmethod

from net import *


class BaseAlgorithm(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 device=torch.device('cuda')):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

    @abstractmethod
    def update(self):
        pass

    def save(self, save_dir):
        torch.save(self.actor.state_dict(), save_dir + "actor")

    def load(self, load_dir):
        self.actor.load_state_dict(torch.load(load_dir + "actor"))

