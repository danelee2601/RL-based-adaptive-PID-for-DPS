"""
ddpg_simple_type1
- it follows DDPG_updated2.py from 'https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow'
- However, Batch Normalization cannot be applied.
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(1)
import numpy as np


class DDPG(object):
    def __init__(self, sess, nb_observations, nb_actions, actor_lr, critic_lr, tau, gamma, batch_size, memory_size,
                 scale_obs, scale_obs_by_const, scale_reward, ddpg_simple_scaler, hl_a, hl_c, var):
        # given attributes
        self.sess = sess
        self.nb_observations = nb_observations
        self.nb_actions = nb_actions
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.scale_obs = scale_obs
        self.scale_obs_by_const = scale_obs_by_const
        self.scale_reward = scale_reward
        self.ddpg_simple_scaler = ddpg_simple_scaler
        self.var = var
        self.hl_a = hl_a
        self.hl_c = hl_c

        # fixed attributes
        self.training_bn = tf.placeholder(tf.bool)

        # memory
        self.memory = np.zeros((self.memory_size, self.nb_observations * 2 + self.nb_actions + 1), dtype=np.float32)
        self.pointer = 0
        self.len_buffer = 0
        self.n = 0

        self.S = tf.placeholder(tf.float32, [None, self.nb_observations], 's')
        self.S_ = tf.placeholder(tf.float32, [None, self.nb_observations], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S,)
        self.q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - self.tau)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        self.a_loss = - tf.reduce_mean(self.q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.actor_lr).minimize(self.a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + self.gamma * q_
            self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.q)
            #self.td_error = tf.losses.huber_loss(labels=q_target, predictions=self.q)
            self.ctrain = tf.train.AdamOptimizer(self.critic_lr).minimize(self.td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s, apply_noise):
        if self.scale_obs:
            s = (s - self.ddpg_simple_scaler.mu_obs) / self.ddpg_simple_scaler.sigma_obs
        elif self.scale_obs_by_const[0]:
            s = s / self.scale_obs_by_const[1]

        a = self.sess.run(self.a, {self.S: s[np.newaxis, :], self.training_bn: False})[0]
        q_val = self.sess.run(self.q, {self.S: s[np.newaxis, :], self.a: a[np.newaxis, :],
                                       self.training_bn: False})[0][0]

        if apply_noise:
            a = np.clip(np.random.normal(a, self.var), 0, 1)

        return a, q_val

    def learn(self):
        indices = np.random.choice(self.memory_size, size=self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.nb_observations]
        ba = bt[:, self.nb_observations: self.nb_observations + self.nb_actions]
        br = bt[:, -self.nb_observations - 1: -self.nb_observations]
        bs_ = bt[:, -self.nb_observations:]

        # scale
        if self.scale_obs:
            bs = (bs - self.ddpg_simple_scaler.mu_obs) / self.ddpg_simple_scaler.sigma_obs
            bs_ = (bs_ - self.ddpg_simple_scaler.mu_obs) / self.ddpg_simple_scaler.sigma_obs
        elif self.scale_obs_by_const[0]:
            bs = bs / self.scale_obs_by_const[1]
            bs_ = bs_ / self.scale_obs_by_const[1]

        if self.scale_reward:
            br = (br - self.ddpg_simple_scaler.mu_r) / self.ddpg_simple_scaler.sigma_r

        _, loss_actor = self.sess.run([self.atrain, self.a_loss], {self.S: bs, self.training_bn: True})
        _, loss_critic = self.sess.run([self.ctrain, self.td_error], {self.S: bs, self.a: ba, self.R: br, self.S_: bs_,
                                                                      self.training_bn: True})
        return loss_actor, loss_critic

    def store_transition(self, s, a, r, s_, update_mu_sigma):
        if update_mu_sigma:
            self.ddpg_simple_scaler.update_mu_sigma_obs(self.len_buffer+1, s)
            self.ddpg_simple_scaler.update_mu_sigma_r(self.len_buffer+1, r)

        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_size  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        self.len_buffer = self.len_buffer + 1 if self.pointer < self.memory_size else self.memory_size

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, self.hl_a, activation=tf.nn.relu, name='l1', trainable=trainable)
            net = tf.layers.dense(net, self.hl_a, activation=tf.nn.relu, name='l2', trainable=trainable)
            a = tf.layers.dense(net, self.nb_actions, activation=tf.nn.sigmoid, name='a1', trainable=trainable)
            return a

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            w1_s = tf.get_variable('w1_s', [self.nb_observations, self.hl_c], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.nb_actions, self.hl_c], trainable=trainable)
            b1 = tf.get_variable('b1', [1, self.hl_c], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, self.hl_c, activation=tf.nn.relu, trainable=trainable)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
