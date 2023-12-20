import torch
import gym
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

from utils.wrap_env import make_env
from utils.data_utils import make_buffer
from utils.logger import Logger, make_log_dirs
from algorithm import *
from trainer import *
from param import *


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")

    expert_path = os.path.join(args.expert_dir, args.demo)
    expert_buffer, policy_buffer, replay_buffer, state_mean, state_std = \
        make_buffer(expert_path, args.subsampling_num, args.subsampling_rate, args.use_absorb, args.truncated_len,
                    device)

    train_env = make_env(args.env_id, args.use_absorb, state_mean, state_std, args.truncated_len, args.seed)
    eval_env = make_env(args.env_id, args.use_absorb, state_mean, state_std, args.truncated_len, args.seed + 1)

    # logger
    log_dirs = make_log_dirs(args.log_dir, args.env_id, args.algo, args.seed, vars(args))
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "pretraining_progress": "csv",
        "finetuning_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))
    
    if args.algo == 'bc':
        algo = BC(state_dim=train_env.observation_dim,
                  action_dim=train_env.action_dim,
                  actor_lr=args.actor_lr,
                  device=device)
    elif args.algo == 'dac':
        algo = DAC(state_dim=train_env.observation_dim,
                   action_dim=train_env.action_dim,
                   discriminator_lr=args.discriminator_lr,
                   critic_lr=args.critic_lr,
                   actor_lr=args.actor_lr,
                   alpha_lr=args.alpha_lr,
                   lamda=args.lamda,
                   ensemble_num=args.ensemble_num,
                   init_temp=args.init_temp,
                   learn_temp=args.learn_temp,
                   device=device)
    elif args.algo == 'value_dice':
        algo = ValueDICE(state_dim=train_env.observation_dim,
                         action_dim=train_env.action_dim,
                         critic_lr=args.critic_lr,
                         actor_lr=args.actor_lr,
                         device=device)
    elif args.algo == 'iq_learn':
        algo = IQLearn(state_dim=train_env.observation_dim,
                       action_dim=train_env.action_dim,
                       critic_lr=args.critic_lr,
                       actor_lr=args.actor_lr,
                       alpha_lr=args.alpha_lr,
                       device=device)
    else:
        raise NotImplementedError

    if args.algo == "bc":
        trainer = OfflineRLTrainer(train_env, eval_env, expert_buffer, policy_buffer, logger)
        trainer.train(algo=algo,
                      num_epoch=args.num_epoch,
                      log_interval=args.log_interval)
    else:      
        trainer = OnlineRLTrainer(train_env, eval_env, expert_buffer, policy_buffer, replay_buffer, logger)
        trainer.train(algo=algo,
                      num_epoch=args.num_epoch,
                      random_epoch=args.random_epoch,
                      start_epoch=args.start_epoch,
                      absorbing_per_episode=args.absorbing_per_episode,
                      update_per_step=args.update_per_step,
                      log_interval=args.log_interval)


if __name__ == "__main__":
    train(args)

