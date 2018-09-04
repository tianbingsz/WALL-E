"""
    Author: Tianbing Xu (xutianbing@baidu.com)
    Created on: 2018-08-03

    Copyright (c) Baidu.com, Inc. All Rights Reserved
"""
import tensorflow as tf
import numpy as np
import scipy.signal

class Sampler(object):
    """
        Generate rollouts of transistions by exec current policy in Env
    """

    def __init__(self,
                 env,
                 policy,
                 scaler,
                 max_step=1000,
                 batch_size=10000,
                 animate=False):
        """
            env: ai gym environment
            policy: MLP policy network
            scaler: scaler object, used to scale/offset each observation
                    dimension to a similar range
            max_step: maximum sample size of a episode
            batch_size: number of steps in a training batch
            animate: boolean, True uses env.render() method to animate episode
        """
        self.env = env
        self.policy = policy
        self.scaler = scaler
        self.max_step = max_step
        self.batch_size = batch_size
        self.animate = animate

    def gen_rollouts(self, sess):
        """ Run policy and collect data for a minimum of min_steps and min_episodes
        Args:
            sess: session to run
        Returns: list of trajectory dictionaries, list length = number of episodes
            'observes' : NumPy array of states from episode
            'actions' : NumPy array of actions from episode
            'rewards' : NumPy array of (un-discounted) rewards from episode
            'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
        """
        total_steps = 0
        trajectories = []
        while total_steps < self.batch_size:
            observes, actions, rewards, unscaled_obs = self.gen_rollout(sess)
            total_steps += observes.shape[0]
            trajectory = {'observes' : observes,
                          'actions' : actions,
                          'rewards' : rewards,
                          'unscaled_obs' : unscaled_obs}
            trajectories.append(trajectory)
        unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
        # update the running stats
        self.scaler.update(unscaled)
        return trajectories

    def gen_rollout(self, sess):
        """ Run single episode with option to animate
        Args:
            sess: session to run
        Returns: 4-tuple of NumPy arrays
            observes: shape = (episode len, obs_dim)
            actions: shape = (episode len, act_dim)
            rewards: shape = (episode len,)
            unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
        """
        obs = self.env.reset()
        observes, actions, rewards, unscaled_obs = [], [], [], []
        done = False
        step = 0.0
        # not scale and offset time step feature
        scale, offset = self.scaler.get()
        scale[-1] = 1.0
        offset[-1] = 0.0

        for _ in range(self.max_step):
            if self.animate:
                env.render()
            obs = obs.astype(np.float32).reshape((1, -1))
            obs = np.append(obs, [[step]], axis=1) # add time step feature
            unscaled_obs.append(obs)
            obs= (obs - offset) * scale
            observes.append(obs)
            action = self.sample(sess, obs).reshape((1, -1)).astype(np.float32)
            actions.append(action)
            obs, reward, done, _ = self.env.step(np.squeeze(action, axis=0))
            if not isinstance(reward, float):
                reward = np.asscalar(reward)
            rewards.append(reward)
            step += 1e-3  # increment time step feature
            if done: break
        return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))

    def sample(self, sess, obs):
        """Draw sample from policy distribution
        Args:
            sess: session to run
            obs: input observations
        """
        feed_dict = {self.policy.obs_ph: obs}
        return sess.run(self.policy.sampled_act, feed_dict=feed_dict)
