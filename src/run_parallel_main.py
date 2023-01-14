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
    # Create environment
    env = GymEnv(env_name, max_step=max_step, animate=animate)
    # Create policy
    policy = Policy(env.obs_dim, env.act_dim, hid1_mult=hid1_mult,
                    log_std=policy_logvar)
    # Create agent
    agent = PPO(policy, env, gamma=gamma, lam=lam, kl_targ=kl_targ,
                epochs=epochs, batch_size=batch_size)
    # Create sampler
    sampler = ParallelSampler(n_sampler, env_name, max_step=max_step)
    # Create logger
    logger = Logger(logname='ppo', logdir='./log/')
    # Create saver
    saver = tf.train.Saver(max_to_keep=5)
    # Create session
    sess = tf.Session()
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    # Start sampler
    sampler.start()
    # Start training
    total_steps = 0
    for i in range(num_iteration):
        # Sample rollouts
        print('Sampling...')
        start_time = time.time()
        rollout = sampler.sample(agent, sess)
        print('Sampled in %.3f seconds' % (time.time() - start_time))
        # Update policy
        print('Updating policy...')
        start_time = time.time()
        agent.update(rollout, sess)
        print('Updated in %.3f seconds' % (time.time() - start_time))
        # Log
        total_steps += rollout['ep_len'].sum()
        logger.log_tabular('Iteration', i)
        logger.log_tabular('TotalSteps', total_steps)
        logger.log_tabular('EpLenMean', rollout['ep_len'].mean())
        logger.log_tabular('EpRewMean', rollout['ep_rew'].mean())
        logger.log_tabular('KL', agent.kl)
        logger.log_tabular('Entropy', agent.entropy)
        logger.log_tabular('ClipFrac', agent.clipfrac)
        logger.log_tabular('StopIter', agent.stop_iter)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()

        if i % 10 == 0:
            saver.save(sess, './log/ppo/model.ckpt', global_step=i)

    # Stop sampler
    sampler.stop()
    # Close session
    sess.close()

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
