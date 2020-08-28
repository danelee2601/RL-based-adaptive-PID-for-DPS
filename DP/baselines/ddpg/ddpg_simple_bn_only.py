"""
ddpg_simple_type2
- it follows DDPG_updated1.py from 'https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow'
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


class DDPG(object):
    def __init__(self, sess, nb_observations, nb_actions, actor_lr, critic_lr, tau, gamma, batch_size, memory_size,
                 scale_obs, scale_obs_by_const, scale_reward, ddpg_simple_scaler, var, hl_a, hl_c, use_bn):
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
        self.use_bn = use_bn

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

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            self.q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - self.tau) * ta + self.tau * ea), tf.assign(tc, (1 - self.tau) * tc + self.tau * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        # loss func
        q_target = self.R + self.gamma * q_
        self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.q)
        #self.td_error = tf.losses.huber_loss(labels=q_target, predictions=self.q)  # For learning stability

        # optimizer
        self.ctrain = tf.train.AdamOptimizer(self.critic_lr).minimize(self.td_error, var_list=self.ce_params)
        self.a_loss = - tf.reduce_mean(self.q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.actor_lr).minimize(self.a_loss, var_list=self.ae_params)

        # global_variables_initializer
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s, apply_noise):
        if self.scale_obs:
            s = (s - self.ddpg_simple_scaler.mu_obs) / self.ddpg_simple_scaler.sigma_obs
        elif self.scale_obs_by_const[0]:
            s = s/self.scale_obs_by_const[1]

        a = self.sess.run(self.a, {self.S: s[np.newaxis, :], self.training_bn: False})[0]
        q_val = self.sess.run(self.q, {self.S: s[np.newaxis, :], self.a: a[np.newaxis, :],
                                       self.training_bn: False})[0][0]

        if apply_noise:
            a = np.clip(np.random.normal(a, self.var), 0, 1)
        return a, q_val

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(self.memory_size, size=self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.nb_observations]
        ba = bt[:, self.nb_observations: self.nb_observations + self.nb_actions]
        br = bt[:, -self.nb_observations - 1: -self.nb_observations]
        bs_ = bt[:, -self.nb_observations:]

        # scale
        if self.scale_obs:
            # for positions
            bs[:, :-3] = (bs[:, :-3] - self.ddpg_simple_scaler.mu_obs[:-3]) / self.ddpg_simple_scaler.sigma_obs[:-3]
            bs_[:, :-3] = (bs_[:, :-3] - self.ddpg_simple_scaler.mu_obs[:-3]) / self.ddpg_simple_scaler.sigma_obs[:-3]

            # for accumulated_errs
            # 이렇게 하는 이유는, 위의 정규화방식을 취하면 트레이닝 동안 처음부터 끝까지 mean/std이 계~속 변해서 값의 변동이 계속해서 일어나게되어,
            # 정작 test때는 똑같은 성능이 안나온다. 그 이유는 test시에는 트레이닝시 마지막 스텝에서의 mean/std을 사용하기 때문.
            bs[:, -3:] = bs[:, -3:] / 100
            bs_[:, -3:] = bs_[:, -3:] / 100

        elif self.scale_obs_by_const[0]:
            bs = bs / self.scale_obs_by_const[1]
            bs_ = bs_ / self.scale_obs_by_const[1]

        if self.scale_reward:
            br = (br - self.ddpg_simple_scaler.mu_r) / self.ddpg_simple_scaler.sigma_r
            #br = (br - 0.) / self.ddpg_simple_scaler.sigma_r

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

    def add_bn_layer(self, net, use_bn, trainable, name):
        if use_bn:
            net = tf.layers.batch_normalization(net, name=name, training=self.training_bn, trainable=trainable)
        return net

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, self.hl_a, activation=tf.nn.relu, name='l1', trainable=trainable)
            #net = self.add_bn_layer(net, self.use_bn, trainable, name='bnl1_a')
            net = tf.layers.dense(net, self.hl_a, activation=tf.nn.relu, name='l2', trainable=trainable)
            #net = self.add_bn_layer(net, self.use_bn, trainable, name='bnl2_a')
            a = tf.layers.dense(net, self.nb_actions, activation=tf.nn.sigmoid, name='a', trainable=trainable)
            return a

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            w1_s = tf.get_variable('w1_s', [self.nb_observations, self.hl_c], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.nb_actions, self.hl_c], trainable=trainable)
            b1 = tf.get_variable('b1', [1, self.hl_c], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = self.add_bn_layer(net, self.use_bn, trainable, name='bnl1_c')  # gradient exploding is caused.
            net = tf.layers.dense(net, self.hl_c, activation=tf.nn.relu, trainable=trainable)
            net = self.add_bn_layer(net, self.use_bn, trainable, name='bnl2_c')
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
