"""
    Author: Tianbing Xu (xutianbing@baidu.com)
    Created on: 2018-08-15

    Copyright (c) Baidu.com, Inc. All Rights Reserved
"""
#! /usr/bin/env python3

import gym
from policy.policy import Policy
from agent.ppo import PPO
from env.sampler import Sampler
from util.utils import Logger, Scaler
from datetime import datetime
import os
import argparse
import signal

class GracefulKiller:
    """ Gracefully exit program on CTRL-C """
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True

def init_gym(env_name):
    """
    Initialize gym environment, return dimension of observation
    and action spaces.
    Args:
        env_name: str environment name (e.g. "Humanoid-v1")
    Returns: 3-tuple
        gym environment (object)
        number of observation dimensions (int)
        number of action dimensions (int)
    """
    env = gym.make(env_name)
    # add 1 to obs dimension for time step feature
    obs_dim = env.observation_space.shape[0] + 1
    act_dim = env.action_space.shape[0]
    return env, obs_dim, act_dim

def main(env_name,
         num_iteration,
         gamma,
         lam,
         batch_size,
         max_step,
         kl_targ,
         hid1_mult,
         policy_logvar,
         epochs,
         animate=False):
    """ Main training loop
    Args:
        env_name: OpenAI Gym environment name, e.g. 'Hopper-v2'
        num_iteration: number of total iterations
        gamma: reward discount factor (float)
        lam: lambda from Generalized Advantage Estimate
        batch_size: number of samples per policy training batch
        max_step: maximum time step each episode
        kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
        hid1_mult: hid1 size for policy and value_f (mutliplier of obs dimension)
        policy_logvar: natural log of initial policy variance
        epochs: num of mini-batch iterations
        animate: boolean, True uses env.render() method to animate episode
    """
    killer = GracefulKiller()
    env, obs_dim, act_dim = init_gym(env_name)
    now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
    logger = Logger(logname="SingleProc"+env_name, now=now)
    aigym_path = os.path.join('log-files/gym', env_name, now)
    env=gym.wrappers.Monitor(env, aigym_path, force=True, video_callable=False)
    scaler = Scaler(obs_dim)

    policy_size = "large"
    if env_name in ['Humanoid-v2', 'HumanoidStandup-v2', 'Ant-v2']:
        policy_size = "small"
    policy = Policy(obs_dim, act_dim, kl_targ, hid1_mult,
                    policy_logvar, policy_size)
    # sampler to generate rollouts by exex policy on Env
    sampler = Sampler(env, policy, scaler, max_step, batch_size, animate)
    # agent PPO to update policy and generate rollout alteratively
    ppo = PPO(policy, sampler, logger, killer,
              num_iteration, epochs, gamma, lam, max_step)
    # agent policy learning for num_iteration
    ppo.train()
    logger.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = ('PPO for OpenAI Gym Envs'))

    parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')
    parser.add_argument('-e', '--epochs', type=float, help='num of mini-batch iterations',
                        default=20)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor',
                        default=0.995)
    parser.add_argument('-mx', '--max_step', type=int, help='Max time step for each episode',
                        default=1000)
    parser.add_argument('-it', '--num_iteration', type=int, help='Number of Iteration for running',
                        default=2000)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for GAE',
                        default=0.98)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default=0.003)
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Number of samples per training batch',
                        default=10000)
    parser.add_argument('-m', '--hid1_mult', type=int,
                        help='first hidden layer size for value and policy network',
                        default=10)
    parser.add_argument('-v', '--policy_logvar', type=float,
                        help='initial policy log-variance',
                        default=-1.0)
    args = parser.parse_args()
    main(**vars(args))
