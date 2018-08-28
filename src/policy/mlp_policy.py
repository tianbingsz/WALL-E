"""
    Author: Tianbing Xu (xutianbing@baidu.com)
    Created on: 2018-08-23

    Copyright (c) Baidu.com, Inc. All Rights Reserved
"""
import numpy as np
import tensorflow as tf
from optimizer.opt_adam import AMSGrad
from util.utils import fully_connected

class Policy(object):
    """ NN-based policy approximation """
    def __init__(self,
                 obs_dim,
                 act_dim,
                 kl_targ,
                 hid1_mult,
                 policy_logvar,
                 policy_size,
                 scope="policy"):
        """
        Args:
            obs_dim: num observation dimensions (int)
            act_dim: num action dimensions (int)
            kl_targ: target KL divergence between pi_old and pi_new
            hid1_mult: size of first hidden layer, multiplier of obs_dim
            policy_logvar: natural log of initial policy variance
            policy_size: large or small network
            scope: policy network variable scope
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.kl_targ = kl_targ
        self.hid1_mult = hid1_mult
        self.policy_logvar = policy_logvar
        self.policy_size = policy_size
        self.scope = scope
        self.lr = None
        self._build_symbolic_graph()

    def _build_symbolic_graph(self):
        """ build ops and computation graph """
        self._placeholders()
        self._policy_nn()
        self._logprob()
        self._kl_entropy()
        self._loss_train_op()

    def _placeholders(self):
        """ Input placeholders """
        # observations, actions and advantages:
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
        self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')
        # strength of D_KL loss terms:
        self.beta_ph = tf.placeholder(tf.float32, (), 'beta')
        self.eta_ph = tf.placeholder(tf.float32, (), 'eta')
        # learning rate
        self.lr_ph = tf.placeholder(tf.float32, (), 'eta')
        # log_vars and means with pi_old (previous step's policy parameters):
        self.old_log_vars_ph = tf.placeholder(tf.float32, (self.act_dim,), 'old_log_vars')
        self.old_means_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means')

    def _policy_nn(self):
        """ Neural net for policy approximation function
            Policy parameterized by Gaussian means and variances. NN outputs mean
            action based on observation. Trainable variables hold log-variances
            for each action dimension (i.e. variances not determined by NN).
        """
        if self.policy_size == 'small':
            print("using small structure")
            hid1_size = self.obs_dim
            hid3_size = self.act_dim
            hid2_size = int(np.sqrt(hid1_size * hid3_size))

        elif self.policy_size == 'large':
            print('Using large structure ')
            hid1_size = self.obs_dim * self.hid1_mult
            hid3_size = self.act_dim  * 10
            hid2_size = int(np.sqrt(hid1_size * hid3_size))
        else:
            raise NotImplementedError

        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        self.lr = 9e-4 / np.sqrt(hid2_size)  # 9e-4 empirically determined
        weight_init = tf.random_uniform_initializer(-0.05, 0.05)
        bias_init = tf.constant_initializer(0)
        # 3 hidden layers with tanh activations
        with tf.variable_scope(self.scope):
            h1 = fully_connected(self.obs_ph, self.obs_dim, hid1_size,
                weight_init, bias_init, "policy_h1")
            h1 = tf.nn.tanh(h1)
            h2 = fully_connected(h1, hid1_size, hid2_size, weight_init,
                bias_init, "policy_h2")
            h2 = tf.nn.tanh(h2)
            h3 = fully_connected(h2, hid2_size, hid3_size, weight_init,
                bias_init, "policy_h3")
            self.means = fully_connected(h3, hid3_size, self.act_dim, weight_init,
                bias_init, "policy_mean")
            # logvar_speed is used to 'fool' gradient descent into making faster updates
            # to log-variances. heuristic sets logvar_speed based on network size.
            logvar_speed = (10 * hid3_size) // 48
            log_vars = tf.get_variable("policy_logvars", (logvar_speed, self.act_dim),
                tf.float32, tf.constant_initializer(0.0))
            self.log_vars = tf.reduce_sum(log_vars, axis=0) + self.policy_logvar
        print('Policy Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}, logvar_speed: {}'
              .format(hid1_size, hid2_size, hid3_size, self.lr, logvar_speed))

    def _logprob(self):
        """ Calculate log probabilities of a batch of observations & actions
            Calculates log probabilities using previous step's model parameters
            and new parameters being trained.
        """
        logp = -0.5 * tf.reduce_sum(self.log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) /
                                     tf.exp(self.log_vars), axis=1)
        self.logp = logp

        logp_old = -0.5 * tf.reduce_sum(self.old_log_vars_ph)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.old_means_ph) /
                                         tf.exp(self.old_log_vars_ph), axis=1)
        self.logp_old = logp_old

    def _kl_entropy(self):
        """
        Add to Graph:
            1. KL divergence between old and new distributions
            2. Entropy of present policy given states and actions
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Entropy
        """
        log_det_cov_old = tf.reduce_sum(self.old_log_vars_ph)
        log_det_cov_new = tf.reduce_sum(self.log_vars)
        tr_old_new = tf.reduce_sum(tf.exp(self.old_log_vars_ph - self.log_vars))

        self.kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                                       tf.reduce_sum(tf.square(self.means - self.old_means_ph) /
                                                     tf.exp(self.log_vars), axis=1) -
                                       self.act_dim)
        self.entropy = 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) +
                              tf.reduce_sum(self.log_vars))

    def _loss_train_op(self):
        """
        Three loss terms:
            1) standard policy gradient
            2) D_KL(pi_old || pi_new)
            3) Hinge loss on [D_KL - kl_targ]^2
        See: https://arxiv.org/pdf/1707.02286.pdf
        """
        loss1 = -tf.reduce_mean(self.advantages_ph *
                                tf.exp(self.logp - self.logp_old))
        loss2 = tf.reduce_mean(self.beta_ph * self.kl)
        loss3 = self.eta_ph * tf.square(tf.maximum(0.0, self.kl - 2.0 * self.kl_targ))
        self.loss = loss1 + loss2 + loss3
        optimizer = AMSGrad(alpha=self.lr_ph)
        self.train_op = optimizer.minimize(self.loss)
