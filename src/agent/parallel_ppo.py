"""
    Author: Tianbing Xu (xutianbing@baidu.com)
    Created on: 2018-08-24

    Copyright (c) Baidu.com, Inc. All Rights Reserved
"""
import tensorflow as tf
import numpy as np
import time
from multiprocessing import Process, Value
from util.sampler_util import process
from util.value_function import NNValueFunction
from util.utils import GetPolicyWeights

class PPO(Process):
    """
        Proximal Policy Gradient with MultiProcessing Support
        For simplicity, we only run one agent process, to communicate with
        multiple sampler processes asynchronously.
        1. Agent waits for the rollouts generated from multi-sampler processes,
        and update the policy based on the rollouts reading from Queue
        2. After updating policy, agent writes new policy into Queue, and multi-sampler
        processes read updated policy weights from Queue, which are used to generate
        rollouts in the next iteration
    """

    def __init__(self,
                 task_q,
                 result_q,
                 policy,
                 logger,
                 num_iteration=2000,
                 epochs=20,
                 gamma=0.995,
                 lam=0.98,
                 max_step=1000):
        """
        Args:
            task_q: Queue for task commands and rollouts
            result_q: Queue for policy weights communication
            policy: MLP policy network
            logger: logging stats to file and out
            num_iteration: total number of running iterations
            epochs: num of mini-batch iterations
            gamma: reward discount factor (float)
            lam: lambda from Generalized Advantage Estimate
            max_step: maximum sample size of a episode
        """
        Process.__init__(self)
        self.task_q = task_q
        self.result_q = result_q

        self.policy = policy
        self.logger = logger
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
        self._get_policy_property()
        # Q value func
        self.val_func = NNValueFunction(
            self.policy.obs_dim, self.policy.hid1_mult)
        # total running time
        self.total_time = Value('d', 0.0)

    def _get_policy_property(self):
        """
            reference the policy's placeholder and ops
        """
        self.scope = self.policy.scope
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

    def run(self):
        """
            inherit from run of Process, keep agent running async
        """
        print('----- agent training -----')
        self.init_session()
        var_list = tf.trainable_variables(self.scope)
        self.get_policy = GetPolicyWeights(self.sess, var_list)
        self.n_episode = 0
        while True:
            rollouts = self.task_q.get()
            if rollouts is None: # kill agent
                print("agent ppo exit")
                self.task_q.task_done()
                break
            elif rollouts == 1:
                # get policy weights, no learning
                self.task_q.task_done()
                self.result_q.put(self.get_policy())
            #TODO, rollouts == 2, save model
            else: # update policy, get policy, save to Queue
                print("update policy, save to Queue")
                self.train_one_epoch(rollouts)
                self.task_q.task_done()
                self.result_q.put(self.get_policy())
        self.close_session()
        return

    def train_one_epoch(self, rollouts):
        """
            train ppo agent for one epoch,
            update policy given rollouts
        """
        self.n_episode += len(rollouts)
        (obs, actions, adv, rewards) = process(
            self.sess, rollouts, self.logger, self.val_func,
            self.gamma, self.lam, self.n_episode)
        # optimize policy with trajectories samples
        self.optimize_policy(self.sess, obs, actions, adv)
        # fit value function
        self.val_func.fit(self.sess, obs, rewards, self.logger)
        self.logger.log(
            {'_MeanReward' : np.mean([t['rewards'].sum() for t in rollouts]),
            'Steps' : self.n_episode * self.max_step,
            'Time' : self.total_time.value}
        )
        # write logger results to file and out
        self.logger.write(display=True)

    def optimize_policy(self, sess, observes, actions, advantages):
        """
            run optimizer to update policy network parameters
        Args:
            sess: session to run
            observes: observations, shape = (N, obs_dim)
            actions: actions, shape = (N, act_dim)
            advantages: advantages, shape = (N,)
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
                                                feed_dict=feed_dict)
        feed_dict[self.old_log_vars_ph] = old_log_vars_np
        feed_dict[self.old_means_ph] = old_means_np
        print(old_log_vars_np)
        loss, kl, entropy = 0, 0, 0
        for e in range(self.epochs):
            # TODO: need to improve data pipeline - re-feeding data every epoch
            sess.run(self.train_op, feed_dict)
            loss, kl, entropy = sess.run([self.loss, self.kl, self.entropy],
                                        feed_dict=feed_dict)
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

    def set_total_time(self, total_time):
        #TODO, add lock for multi-proc
        self.total_time.value = total_time

    @property
    def value(self):
        return self.total_time.value

    def init_session(self):
        """ init session to run ops in agent process """
        self.sess = tf.Session(config=tf.ConfigProto(
            device_count = {'GPU': 0}))
        self.sess.run(tf.global_variables_initializer())

    def close_session(self):
        self.sess.close()

    def exit(self):
        """ kill agent process """
        self.task_q.put(None)
