"""
    Author: Tianbing Xu (xutianbing@baidu.com)
    Created on: 2018-08-02

    Copyright (c) Baidu.com, Inc. All Rights Reserved
"""
import tensorflow as tf
import numpy as np
import time
from util.sampler_util import process
from util.value_function import NNValueFunction

class PPO(object):
    """
        Proximal Policy Gradient
    """

    def __init__(self,
                 policy,
                 sampler,
                 logger,
                 killer,
                 num_iteration=2000,
                 epochs=20,
                 gamma=0.995,
                 lam=0.98,
                 max_step=1000):
        """
        Args:
            policy: MLP policy
            sampler: to generate rollouts by exec policy on MDP
            killer: GracefulKiller
            num_iteration: total number of running iterations
            epochs: num of mini-batch iterations
            gamma: reward discount factor (float)
            lam: lambda from Generalized Advantage Estimate
            max_step: maximum sample size of a episode
        """
        self.policy = policy
        self.sampler = sampler
        self.logger = logger
        self.killer = killer
        # hyper-parameter
        self.num_iteration = num_iteration
        self.epochs = epochs
        self.beta = 1.0  # dynamically adjusted D_KL loss multiplier
        self.eta = 50  # multiplier for D_KL-kl_targ hinge-squared loss
        self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self.gamma = gamma
        self.lam = lam
        self.max_step = max_step
        # policy related
        self._set_policy_ops()
        # Q value func
        self.val_func = NNValueFunction(
            self.policy.obs_dim, self.policy.hid1_mult)

    def _set_policy_ops(self):
        """
            reference the policy's placehoer and ops
        """
        # policy placeholer
        self.obs_ph = self.policy.obs_ph
        self.act_ph = self.policy.act_ph
        self.advantages_ph = self.policy.advantages_ph
        self.beta_ph = self.policy.beta_ph
        self.eta_ph = self.policy.eta_ph
        self.lr_ph = self.policy.lr_ph
        self.old_log_vars_ph = self.policy.old_log_vars_ph
        self.old_means_ph = self.policy.old_means_ph
        # policy learning rate
        self.lr = self.policy.lr
        # policy ops
        self.means = self.policy.means
        self.log_vars = self.policy.log_vars
        self.train_op = self.policy.train_op
        self.loss = self.policy.loss
        self.kl = self.policy.kl
        self.kl_targ = self.policy.kl_targ
        self.entropy = self.policy.entropy

    def init_session(self):
        self.sess = tf.Session(config=tf.ConfigProto(
            device_count = {'GPU': 0}))
        self.sess.run(tf.global_variables_initializer())

    def close_session(self):
        self.sess.close()

    def train(self):
        """
            train ppo agent
            while True:
                data = gen_rollouts(policy)
                policy = update_policy(data)
        """
        self.init_session()
        print('----- agent training -----')
        n_episode = 0
        start_time = time.time()
        for iter in range(self.num_iteration):
            # gen rollouts by exec policy on MDP
            rollout_start = time.time()
            rollouts = self.sampler.gen_rollouts(self.sess)
            rollout_time = (time.time() - rollout_start) / 60.0

            n_episode += len(rollouts)
            learn_start = time.time()
            (obs, actions, adv, rewards) = process(self.sess,
                rollouts, self.logger, self.val_func,
                self.gamma, self.lam, n_episode)
            # optimize policy with rollouts samples
            self.optimize_policy(self.sess, obs, actions, adv)
            # fit value function
            self.val_func.fit(self.sess, obs, rewards, self.logger)
            learn_time = (time.time() - learn_start) / 60.0

            total_time = (time.time() - start_time) / 60.0
            # write logger results to file and out
            self.logger.log(
                {'_MeanReward' : np.mean([t['rewards'].sum() for t in rollouts]),
                'Steps' : n_episode * self.max_step,
                'Time' : total_time}
            )
            self.logger.write(display=True)
            print('iter {}, running time {}, rollout {}, learn {} min'.format(
                iter, total_time, rollout_time, learn_time))
            if self.killer.kill_now:
                if input('Terminate training (y/[n])? ') == 'y':
                    break
                self.killer.kill_now = False
        self.close_session()

    def optimize_policy(self, sess, observes, actions, advantages):
        """
            run optimizer to update policy network parameters
        Args:
            sess: session to run
            observes: observations, shape = (N, obs_dim)
            actions: actions, shape = (N, act_dim)
            advantages: advantages, shape = (N,)
            logger: Logger object, see utils.py
        """
        print('self.lr: {}, self.lr_mult: {}, self.kl_targ {}'.format(
            self.lr, self.lr_multiplier, self.kl_targ))
        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages,
                     self.beta_ph: self.beta,
                     self.eta_ph: self.eta,
                     self.lr_ph: self.lr * self.lr_multiplier}
        old_means_np, old_log_vars_np = sess.run([self.means, self.log_vars],
                                                      feed_dict)
        feed_dict[self.old_log_vars_ph] = old_log_vars_np
        feed_dict[self.old_means_ph] = old_means_np
        print(old_log_vars_np)
        loss, kl, entropy = 0, 0, 0
        for e in range(self.epochs):
            # TODO: need to improve data pipeline - re-feeding data every epoch
            sess.run(self.train_op, feed_dict)
            loss, kl, entropy = sess.run([self.loss, self.kl, self.entropy], feed_dict)
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # TODO: too many "magic numbers" in next 8 lines of code, need to clean up
        if kl > self.kl_targ * 2:  # servo beta to reach D_KL target
            self.beta = np.minimum(35, 1.5 * self.beta)  # max clip beta
            if self.beta > 30 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2:
            self.beta = np.maximum(1 / 35, self.beta / 1.5)  # min clip beta
            if self.beta < (1 / 30) and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5

        self.logger.log({'PolicyLoss': loss,
                    'PolicyEntropy': entropy,
                    'KL': kl,
                    'Beta': self.beta,
                    '_lr_multiplier': self.lr_multiplier})
