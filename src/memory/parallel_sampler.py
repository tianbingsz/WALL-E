"""
    Author: Tianbing Xu (xutianbing@baidu.com)
    Created on: 2018-08-23

    Copyright (c) Baidu.com, Inc. All Rights Reserved
"""
import tensorflow as tf
import gym
import numpy as np
from random import randint
import scipy.signal
import string
import time
from multiprocessing import Process, Queue, JoinableQueue, Event
from util.utils import SetPolicyWeights, fully_connected, Scaler

class Sampler(Process):
    """
        each sampler process generates rollouts of transistions parallelly by
        exec current policy in independent random seeded Env
    """

    def __init__(self,
                 sid,
                 task_q,
                 result_q,
                 weights_ready_event,
                 env_name,
                 policy,
                 max_step=1000,
                 batch_size=10000,
                 animate=False):
        """
            sid: sampler id for each process
            task_q: Queue for communication of task commands or policy weights
            result_q: Queue for communication of rollouts
            weights_ready_event: clear or set for weights ready
            env_name: ai gym environment, e.g. 'HalfCheetah-V2'
            policy: policy object with ops
            max_step: maximum sample size of a episode
            batch_size: number of steps in a training batch
            animate: boolean, True uses env.render() method to animate episode
        """
        Process.__init__(self)
        self.sid = sid
        self.task_q = task_q
        self.result_q = result_q
        self.weights_ready_event = weights_ready_event

        self.scope = "policy{}".format(self.sid)
        self.env_name = env_name
        self.policy = policy

        self.max_step = max_step
        self.batch_size = batch_size
        self.animate = animate
        self._get_policy_property()
        # used to scale/offset each observation dim to a similar range
        self.scaler = Scaler(self.obs_dim)
        print('proc {} init'.format(sid))

    def _get_policy_property(self):
        """ reference policy properties """
        self.obs_dim = self.policy.obs_dim
        self.act_dim = self.policy.act_dim
        self.hid1_mult = self.policy.hid1_mult
        self.policy_size = self.policy.policy_size
        self.policy_logvar = self.policy.policy_logvar

    def run(self):
        """
            inherit from run of Process, keep sampler running async
        """
        self.env = gym.make(self.env_name)
        self.env.seed(randint(0,999999))
        # TODO, env.monitor
        self._build_policy_nn()
        self._sample()
        self.sess = tf.Session(config=\
            tf.ConfigProto(device_count = {'GPU':0},
            ))
        self.sess.run(tf.global_variables_initializer())
        var_list = tf.trainable_variables(self.scope)
        self.set_policy = SetPolicyWeights(self.sess, var_list)

        while True:
            # get a task, or wait until it gets one
            next_task = self.task_q.get(block=True)
            if next_task == 1:
                # task: request to collect experience
                print('sampler {} gen rollout'.format(self.sid))
                rollouts = self.gen_rollouts(self.sess)
                print('sampler {} finish generating rollouts'.format(self.sid))
                self.task_q.task_done()
                self.result_q.put(rollouts)
            elif next_task == 2:
                print('sampler {} exit'.format(self.sid))
                self.task_q.task_done()
                break
            else: # task: to cache policy parameters
                # assign policy weights of main process from agent's
                # Queue to variables of this sampler process
                self.set_policy(next_task)
                self.task_q.task_done()
                self.weights_ready_event.wait()
        self.sess.close()
        return

    def _sample(self):
        """ Symbolic function, Sample from distribution, given observation """
        self.sampled_act = (self.means +
                            tf.exp(self.log_vars / 2.0) *
                            tf.random_normal(shape=(self.act_dim,)))

    def sample(self, sess, obs):
        """Draw sample from policy distribution
        Args:
            sess: session to run
            obs: input observations
        """
        feed_dict = {self.obs_ph: obs}
        return sess.run(self.sampled_act, feed_dict=feed_dict)

    def _build_policy_nn(self):
        """
        Build ops and computation graph for policy Network
        Policy parameterized by Gaussian means and variances.
        NN outputs mean action based on observation.
        Each sampler process has its own local policy network,
        the weights are assigned from the main process via Queue
        """
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
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
        print('Policy Params in scope {} -- h1: {}, h2: {}, ' + \
              'h3: {}, logvar_speed: {}'.format(
              self.scope, hid1_size, hid2_size, hid3_size, logvar_speed))

    def gen_rollouts(self, sess):
        """ Run policy and collect data for batch_size steps,
            equivalent to batch_size/max_step episodes
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
        # update the running stats,
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
            unscaled_obs: useful for training scaler,
                        shape = (episode len, obs_dim)
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

class ParallelSampler():
    """
        Generate rollouts by N parallel Sampler Processes
        1. Sampler process runs async, generates rollouts by exec policy
        on each own random seeded Env
        2. Sampler writes rollouts to Queue to communicate with Agent process
        3. Sampler reads policy weights from Queue updated by Agent Process
    """

    def __init__(self,
                 n_sampler,
                 env_name,
                 policy,
                 max_step=1000,
                 batch_size=10000,
                 animate=False):
        """
            n_sampler: num of sampler processes to generate rollouts
            env_name: ai gym environment, e.g. 'HalfCheetah-V2'
            policy: policy object with ops
            max_step: maximum sample size of a episode
            batch_size: number of steps in a training batch
            animate: boolean, True uses env.render() method to animate episode
        """
        # Queue and Event
        self.tasks = JoinableQueue()
        self.results = Queue()
        self.weights_ready_event = Event()

        self.policy = policy
        self.n_sampler = n_sampler
        self.batch_size = batch_size
        self.clear_rollouts()

        self.samplers = []
        #TODO, add monitor in one process
        for sid in range(self.n_sampler):
            self.samplers.append(
                Sampler(sid, self.tasks, self.results,
                        self.weights_ready_event, env_name, policy,
                        max_step, batch_size, animate))
        for sampler in self.samplers:
            # each sampler start running async
            sampler.start()

    def set_policy_weights(self, weights):
        """
            save policy weights to tasks Queue,
            signal each sampler process to assign the weights
        """
        self.weights_ready_event.clear()
        for i in range(self.n_sampler):
            self.tasks.put(weights)
        self.tasks.join()
        self.weights_ready_event.set()

    def gen_rollouts(self):
        """
            N Sampler Processes to generate rollouts in parallel,
            for efficiency, each Sampler collects one rollout (episode)
        """
        start = time.time()
        for i in range(self.n_sampler):
            # task to collect experience for each sampler
            self.tasks.put(1)
        # wait for experience collection tasks finishing
        self.tasks.join()

        self.clear_rollouts()
        print('reading result')
        total_steps = 0
        for i in range(self.n_sampler):
            res = self.results.get()
            total_steps += self.add_rollouts(res)
        #print('running {} min to collect total steps {}'.format(
            #(time.time() - start) / 60.0, total_steps))
        return self.rollouts

    def clear_rollouts(self):
        self.rollouts = []

    def add_rollouts(self, to_add):
        """
            add rollouts: self.rollouts += to_add
            self.rollouts: list of rollout (map)
            rollout: map of {'observers' : NumPy array of states from episode
                             'actions' : NumPy array of actions from episode
                             'rewards' : NumPy array of (un-discounted) rewards from episode
                             'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
                             }
            input:
                to_add: list of rollout
            output:
                n_steps in to_add rollouts
        """
        n_steps = 0
        for rollout in to_add:
            self.rollouts.append(rollout)
            n_steps += rollout['observes'].shape[0]
        return n_steps

    def exit(self):
        """
            task: sampler finish
        """
        for i in range(self.n_sampler):
            self.tasks.put(2)
