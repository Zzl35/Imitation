import gym
import numpy as np


def make_env(env_id, use_absorb=True, state_mean=None, state_std=None, max_episode=1000, seed=42):
    if state_mean is None:
        env = TruncatedEnv(env_id, max_episode) if max_episode > 0 else gym.make(env_id)
    else:
        if use_absorb:
            env = AbsorbEnv(env_id, state_mean, state_std, max_episode)
        else:
            env = NormEnv(env_id, state_mean, state_std, max_episode)

    env.seed(seed)
    env.action_space.seed(seed)

    return env


class TruncatedEnv(object):
    def __init__(self, env_id, max_episode=1000):
        self.env = gym.make(env_id)
        self.action_space = self.env.action_space
        self.action_dim = self.env.action_space.shape[0]
        self.observation_dim = self.env.observation_space.shape[0]
        self.max_episode = max_episode
        self.episode = 0

    def seed(self, seed):
        self.env.seed(seed)

    def reset(self):
        state = self.env.reset()
        self.episode = 0
        return state

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.episode += 1
        if self.episode >= self.max_episode:
            done = True
        return next_state, reward, done, info


class NormEnv(TruncatedEnv):
    def __init__(self, env_id, state_mean, state_std, max_episode=1000):
        super().__init__(env_id, max_episode)
        self.state_mean = state_mean
        self.state_std = state_std

        self.reset()

    def normalization(self, state):
        return (state - self.state_mean) / self.state_std

    def reset(self):
        state = self.normalization(self.env.reset())
        self.episode = 0
        return state

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = self.normalization(next_state)
        self.episode += 1
        if self.episode >= self.max_episode:
            done = True
        return next_state, reward, done, info


class AbsorbEnv(NormEnv):
    def __init__(self, env_id, state_mean, state_std, max_episode=1000):
        super().__init__(env_id, state_mean, state_std, max_episode)
        self.observation_dim = self.env.observation_space.shape[0] + 1

    @property
    def absorbing_state(self):
        absorb_state = np.zeros(self.observation_dim, )
        absorb_state[-1] = 1.
        return absorb_state

    @property
    def absorbing_action(self):
        return np.zeros(self.action_dim, )

    def absorbing(self, state):
        state = np.concatenate([state, [0.]])
        return state

    def reset(self):
        state = self.normalization(self.env.reset())
        state = self.absorbing(state)
        self.episode = 0
        return state

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = self.normalization(next_state)
        next_state = self.absorbing(next_state)
        self.episode += 1
        if self.episode >= self.max_episode:
            done = True
        return next_state, reward, done, info


