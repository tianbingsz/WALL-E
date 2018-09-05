"""
    Author: Tianbing Xu (xutianbing@baidu.com)
    Created on: 2018-09-05

    Copyright (c) Baidu.com, Inc. All Rights Reserved
"""
import gym

class GymEnv(object):
    """
        utilities for Gym Env creating, wrapper...
    """

    def __init__(self, env_name="HalfCheetah-v2"):
        self.init_gym(env_name);

    def init_gym(self, env_name):
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
        self.env = gym.make(env_name)
        # add 1 to obs dimension for time step feature
        self.obs_dim = self.env.observation_space.shape[0] + 1
        self.act_dim = self.env.action_space.shape[0]

    def seed(self, rand):
        """
            rand seed env
        """
        self.env.seed(rand)

    def wrapper(self, path, video_callable=False, force=True):
        """gym wrappers with Monitor
           Args:
                env: gym env
                path: logging path
                video_callable: vidoe on/off
            Return : wrapped env
        """
        return gym.wrappers.Monitor(self.env, path, video_callable, force)
