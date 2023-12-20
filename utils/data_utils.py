import numpy as np
import torch
import os, pickle

from utils.replay_buffer import Memory


def load_data(data):
    state, action, next_state, mask = data['states'], data['actions'], data['next_states'], data['dones']
    traj_idx = [0]
    for i in range(len(data['states'])):
        mask[i] = 0. if mask[i] else 1.
        if mask[i] == 0.:
            traj_idx.append(i + 1)
    return state, action, next_state, mask, traj_idx


def add_absorb_state(state, action, next_state, mask, truncated_len=1000):
    ABSORBING_STATE = np.zeros(len(state[0]) + 1, )
    ABSORBING_STATE[-1] = 1.
    ABSORBING_ACTION = np.zeros(len(action[0]))
    prev = -1
    state = np.hstack([state, np.zeros([state.shape[0], 1])])
    next_state = np.hstack([next_state, np.zeros([next_state.shape[0], 1])])
    for i in range(len(state)):
        if not mask[i] and i - prev < truncated_len:
            state = np.vstack([state, ABSORBING_STATE])
            action = np.vstack([action, ABSORBING_ACTION])
            next_state = np.vstack([next_state, ABSORBING_STATE])
            mask = np.vstack([mask.reshape(-1, 1), [-1.]])
            prev = i
    return state, action, next_state, mask


def subsampling_trajectory(state, action, next_state, mask, traj_idx, subsampling_num=1):
    sample_n = np.random.randint(0, len(traj_idx) - 1, subsampling_num)
    sample_ind = []
    for n in sample_n:
        sample_ind.extend([i for i in range(traj_idx[n], traj_idx[n + 1])])
    state = np.array(state)[sample_ind]
    action = np.array(action)[sample_ind]
    next_state = np.array(next_state)[sample_ind]
    mask = np.array(mask)[sample_ind]
    return state, action, next_state, mask


def subsampling_transitions(state, action, next_state, mask, subsampling_rate=20):
    idx = 0
    subsampling_ind = []
    offset = np.random.randint(0, subsampling_rate)
    for i in range(len(state)):
        idx += 1
        if mask[i] == -1. or (idx + offset) % subsampling_rate == 0:
            subsampling_ind.append(i)
        if mask[i] == 0. or mask[i] == -1.:
            idx = 0
            offset = np.random.randint(0, subsampling_rate)
    state = state[subsampling_ind]
    action = action[subsampling_ind]
    next_state = next_state[subsampling_ind]
    mask = mask[subsampling_ind]
    return state, action, next_state, mask


def make_buffer(expert_path,
                subsampling_num=4,
                subsampling_rate=20,
                use_absorb=True,
                truncated_len=1000,
                device=torch.device('cuda')):
    # load trajectory
    assert os.path.isfile(expert_path), f"Not exist {expert_path}"
    with open(expert_path, 'rb') as f:
        trajs = pickle.load(f)
    # flatten = lambda x: x.reshape(x.shape[0]*x.shape[1], *x.shape[2:])
    flatten_convert = lambda items: np.vstack([np.array(items[i]) for i in range(len(items))])
    # compute states, ...
    state = flatten_convert(trajs['states'])
    action = flatten_convert(trajs['actions'])
    next_state = flatten_convert(trajs['next_states'])
    mask = np.where(np.concatenate([np.array(item) for item in trajs['dones']]), 0, 1)
    # compute traj_idx
    traj_lens = np.array(trajs['lengths'])
    traj_idx = np.array([0] + [sum(traj_lens[:i]) for i in range(1, len(traj_lens-1))])

    # subsample trajectory
    state, action, next_state, mask = \
        subsampling_trajectory(state, action, next_state, mask, traj_idx, subsampling_num)

    # normalize states
    state_mean, state_std = np.mean(state, axis=0), np.std(state, axis=0) + 1e-3
    state = (state - state_mean) / state_std
    next_state = (next_state - state_mean) / state_std

    # absorb states
    if use_absorb:
        state, action, next_state, mask = add_absorb_state(state, action, next_state, mask, truncated_len)

    # initialize buffer
    expert_buffer = Memory(memory_size=int(1e5), device=device)
    policy_buffer = Memory(memory_size=int(1e6), device=device)
    replay_buffer = Memory(memory_size=int(1e6), device=device)

    # subsample transitions
    state, action, next_state, mask = subsampling_transitions(state, action, next_state, mask, subsampling_rate)

    # add transitions to buffer
    for i in range(len(state)):
        weight = 1 / subsampling_rate if mask[i] == -1. else 1.
        expert_buffer.add((state[i], action[i], next_state[i], mask[i], weight))
        replay_buffer.add((state[i], action[i], next_state[i], mask[i], weight))

    return expert_buffer, policy_buffer, replay_buffer, state_mean, state_std
