"""
    Author: Tianbing Xu (xutianbing@baidu.com)
    Created on: 2018-08-16

    Copyright (c) Baidu.com, Inc. All Rights Reserved
    Utilities to process rollout data
"""
import numpy as np
import scipy.signal

def process(sess, rollouts, logger, val_func, gamma, lam, n_episode):
    """ Process collection of rollouts
    Args:
        sess : session to run
        rollouts: list of eposide transistions from gen_rollouts
        logger: logging stats to file or out
        val_func: Value Function Q
        gamma, lam: parameter for GAE
        n_episdoe: num of episodes processed
    Returns: 4-tuple of NumPy arrays after processing of rollouts
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    # add estimated Q-values
    add_value(sess, rollouts, val_func)
    # calc discounted sum of Rs, MC Returns
    add_disc_sum_rew(rollouts, gamma)
    # calc advantage
    add_gae(rollouts, gamma, lam)
    # concatenate to Numpy arrays
    observes, actions, advantages, disc_sum_rew = build_train_data(rollouts)
    log_batch_stats(logger, observes, actions, advantages,
                    disc_sum_rew, n_episode)
    return observes, actions, advantages, disc_sum_rew

def build_train_data(rollouts):
    """ building training data based on rollouts
    Args:
        rollouts: rollouts after processing by add_disc_sum_rew(),
            add_value(), and add_gae()
    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes = np.concatenate([t['observes'] for t in rollouts])
    actions = np.concatenate([t['actions'] for t in rollouts])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in rollouts])
    advantages = np.concatenate([t['advantages'] for t in rollouts])
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    return observes, actions, advantages, disc_sum_rew

def add_value(sess, rollouts, val_func):
    """ Adds estimated value to all time steps of all rollouts
    Args:
        sess : session to run
        rollouts: as returned by gen_rollouts()
        val_func: object with predict() method, takes observations
                and returns predicted state value
    Returns:
        None (mutates rollouts dictionary to add 'values')
    """
    for rollout in rollouts:
        observes = rollout['observes']
        values = val_func.predict(sess, observes)
        rollout['values'] = values

def add_gae(rollouts, gamma, lam):
    """ Add generalized advantage estimator.
        https://arxiv.org/pdf/1506.02438.pdf
    Args:
        rollouts: as returned by gen_rollouts(), must include 'values'
                key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
            lam=0 : use TD residuals
            lam=1 : A =  Sum Discounted Rewards - V_hat(s)
    Returns:
        None (mutates rollouts dictionary to add 'advantages')
    """
    for rollout in rollouts:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = rollout['rewards'] * (1 - gamma)
        else:
            rewards = rollout['rewards']
        values = rollout['values']
        # temporal differences
        # tds_t = r_t + v_{t+1} - v_t
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        # adv_t = \sum_{l=0} (gamma *lam)^l tds_{t+l}
        advantages = discount(tds, gamma * lam)
        rollout['advantages'] = advantages

def add_disc_sum_rew(rollouts, gamma):
    """ Adds discounted sum of rewards to all time steps of all rollouts
        R_1 = \sum_{i=1}^n \gamma^{i-1} r_i
        R_k = \sum_{i=k}^n \gamma^{i-1} r_i
    Args:
        rollouts: as returned by gen_rollouts()
        gamma: discount
    Returns:
        None (mutates rollouts dictionary to add 'disc_sum_rew')
    """
    for rollout in rollouts:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = rollout['rewards'] * (1 - gamma)
        else:
            rewards = rollout['rewards']
        disc_sum_rew = discount(rewards, gamma)
        rollout['disc_sum_rew'] = disc_sum_rew

def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point
    R_n = \gamma * R_{n-1} + x_1, R_k = \gamma * R_{k-1} + x_{n+1 -k}
    return [V_1, V_2, ...] = [R_n, R_{n-1}, .... R_1]
    V_n = x_n + \gamma * V_{n-1}
    The first V is MC return, V_1 = R_n = \sum_{k=1}^n \gamma^{k-1} x_{k} """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

def log_batch_stats(logger, observes, actions, advantages, disc_sum_rew, n_episode):
    """ Log various batch statistics """
    logger.log({'_mean_obs': np.mean(observes),
                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.mean(np.var(observes, axis=0)),
                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.mean(np.var(actions, axis=0)),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': n_episode
                })
