"""
    Author: Tianbing Xu (xutianbing@baidu.com)
    Created on: 2018-08-01

    Copyright (c) Baidu.com, Inc. All Rights Reserved
"""
import tensorflow as tf

class Adam(tf.train.Optimizer):
    def __init__(self,
                 alpha=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = {}
        self.u = {}
        self.t = tf.Variable(0.0, trainable=False)
        for v in tf.trainable_variables():
            self.m[v] = tf.Variable(tf.zeros(tf.shape(v.initial_value)), trainable=False)
            self.u[v] = tf.Variable(tf.zeros(tf.shape(v.initial_value)), trainable=False)

    def apply_gradients(self, gvs, global_step=None, name=None):
        t = self.t.assign_add(1.0)
        update_ops = []
        for (g,v) in gvs:
            m = self.m[v].assign(self.beta1 * self.m[v] + (1-self.beta1) * g)
            u = self.u[v].assign(self.beta2 * self.u[v] + (1-self.beta2) * g * g)
            update = -self.alpha * m /(tf.sqrt(u) + self.epsilon)
            update_ops.append(v.assign_add(update))
        return tf.group(*update_ops)

class AMSGrad(tf.train.Optimizer):
    def __init__(self,
                 alpha=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = {}
        self.u = {}
        self.u_hat = {}
        self.t = tf.Variable(0.0, trainable=False)
        for v in tf.trainable_variables():
            self.m[v] = tf.Variable(tf.zeros(tf.shape(v.initial_value)), trainable=False)
            self.u[v] = tf.Variable(tf.zeros(tf.shape(v.initial_value)), trainable=False)
            self.u_hat[v] = tf.Variable(tf.zeros(tf.shape(v.initial_value)), trainable=False)

    def apply_gradients(self, gvs, global_step=None, name=None):
        """
            m_t = beta_1 * m_{t - 1} + (1 - beta_1) * g
            u_t = beta_2 * u_{t - 1} + (1 - beta_2) * g * g
            u_hat_t = max(u_t, u_hat_{t-1})
            w_{t+1} = w_{t} - alpha * m_{t} / sqrt(u_hat_t)
        """
        t = self.t.assign_add(1.0)
        update_ops = []
        for (g,v) in gvs:
            m = self.m[v].assign(self.beta1 * self.m[v] + (1-self.beta1) * g)
            u = self.u[v].assign(self.beta2 * self.u[v] + (1-self.beta2) * g * g)
            u_hat = self.u_hat[v].assign(tf.maximum(self.u_hat[v], u))

            update = -self.alpha * m / (tf.sqrt(u_hat) + self.epsilon)
            update_ops.append(v.assign_add(update))
        return tf.group(*update_ops)
