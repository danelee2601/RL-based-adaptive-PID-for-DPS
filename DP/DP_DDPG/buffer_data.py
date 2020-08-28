import os
import time
from collections import deque
import tensorflow as tf
import numpy as np


class BufferData(object):
    def __init__(self, sess, saver, dps_settings, load_model):
        # given attributes
        self.sess = sess
        self.saver = saver

        # given attributes - class instances
        self.dps_settings = dps_settings

        # set attributes
        self.dirname_models = f'../results/train/{self.dps_settings.dirname}/models'
        self.epoch_start_time = time.time()
        self.epoch_episodes = 0

        # NOTE: All variables must be declared before 'tf.global_variables_initializer()'.
        # restore a trained model
        if load_model and os.listdir(self.dirname_models):
            self.ckpt = tf.train.get_checkpoint_state(self.dirname_models)
            print("\n<< RESTORE >> self.ckpt.model_checkpoint_path : \n{}".format(self.ckpt.model_checkpoint_path))
            saver.restore(sess, self.ckpt.model_checkpoint_path)

            if dps_settings.dps_type == 'openai':
                print('=================================================================================')
                print('sess.run(tf.trainable_variables()[0]) : ', sess.run(tf.trainable_variables()[0]))
                test_obs_ = np.arange(dps_settings.nb_observations)
                test_a, test_q = dps_settings.agent.pi(test_obs_, apply_noise=False, compute_Q=True)  # pi : policy's symbol
                obs_rms_mean, obs_rms_std = self.get_obs_rms_mean_std()
                print('test_obs_ : ', test_obs_)
                print('test_a : ', test_a)
                print('test_q : ', test_q)
                print('obs_rms_mean: ', obs_rms_mean)
                print('obs_rms_std: ', obs_rms_std)
                print('=================================================================================\n')

        # interactive attributes - initialized every simulation
        self.s_list = deque(maxlen=2)
        self.a_list = deque(maxlen=2)
        self.t_print = 0
        self.thrusts = {f'no.{no}': {'Fx': 0., 'Fy': 0., 'F':0., 'Mz': 0.}
                        for no in range(1, self.dps_settings.n_dp + 1)}

    def get_obs_rms_mean_std(self):
        return self.dps_settings.agent.obs_rms_mean, self.dps_settings.agent.obs_rms_std

