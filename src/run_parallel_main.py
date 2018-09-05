"""
    Author: Tianbing Xu (xutianbing@baidu.com)
    Created on: 2018-08-28

    Copyright (c) Baidu.com, Inc. All Rights Reserved
"""
import numpy as np
import tensorflow as tf
import argparse
import json
import os
import time
from datetime import datetime
from util.utils import Logger
from env.gym_env import GymEnv
from memory.parallel_sampler import ParallelSampler
from policy.mlp_policy import Policy
from agent.parallel_ppo import PPO
from multiprocessing import Queue, JoinableQueue

def main(n_sampler,
         env_name,
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
        n_sampler: num of sampler processes to generate rollouts
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
    gymEnv = GymEnv(env_name)
    # create unique directories
    now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")
    logger = Logger(logname='MultProc'+env_name, now=now)
    policy_size = "large"
    if env_name in ['Humanoid-v2', 'HumanoidStandup-v2', 'Ant-v2']:
        policy_size = "small"
    # MLP policy network
    policy = Policy(gymEnv.obs_dim, gymEnv.act_dim, kl_targ, hid1_mult,
            policy_logvar, policy_size)

    # rollouts or agent commands
    agent_tasks = JoinableQueue()
    # communcaiton of policy weights
    agent_results = Queue()
    # PPO Agent Process run async
    agent = PPO(agent_tasks, agent_results, policy, logger,
        num_iteration, epochs, gamma, lam, max_step)
    agent.start()

    # generate rollouts in parallel (by each process)
    pSampler = ParallelSampler(n_sampler, env_name, policy,
                                max_step, batch_size, animate)
    # get init policy weights
    agent_tasks.put(1)
    agent_tasks.join()
    init_weights = agent_results.get()
    pSampler.set_policy_weights(init_weights)

    total_time = 0.0
    for iter in range(num_iteration):
        print("-------- Iteration {} ----------".format(iter))
        agent.set_total_time(total_time)

        # runs a bunch of async processes that collect rollouts
        print("-------- Generate rollouts in Parallel ------")
        rollout_start = time.time()
        rollouts = pSampler.gen_rollouts()
        rollout_time = (time.time() - rollout_start) / 60.0

        # agent receive rollouts and update policy async
        learn_start = time.time()
        # save rollouts generating from parallel samplers
        agent_tasks.put(rollouts)
        agent_tasks.join()
        learn_time = (time.time() - learn_start) / 60.0

        # read policy weights from agent Queue
        print("-------- Get policy weights from Agent ------")
        new_policy_weights = agent_results.get()
        #TODO, save policy weights, calc totalsteps
        print("-------- Update policy weights to Samplers -----\n\n")
        pSampler.set_policy_weights(new_policy_weights)
        total_time += (time.time() - rollout_start) / 60.0
        print("Total time: {}  mins, Rollout time: {}, Learn time: {}".format(
            total_time, rollout_time, learn_time))

    logger.close()
    # exit parallel sampler
    pSampler.exit()
    #TODO, save policy weights
    # exit ppo agent
    agent.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel RL Framework (PPO)')
    parser.add_argument('-n', '--n_sampler', type=int,
                        help='number of samplers to collect experience',
                        default=1)
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
