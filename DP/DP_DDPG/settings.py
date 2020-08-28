import os
import time as time
import shutil
from collections import deque

import numpy as np
import pandas as pd
import OrcFxAPI
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(1)
import dp_force_windff  # included in Lib (.pyd is in DP/dp_force_source/rev03)

from baselines.ddpg.ddpg_simple import DDPG as DDPGSimple
from baselines.ddpg.ddpg_simple_bn_only import DDPG as DDPGSimple_bn_only
from baselines.ddpg.ddpg_simple_scaler import DdpgSimpleScaler
import baselines.common.tf_util as U


class DPSSettings(object):

    def __init__(self):
        """
        dps_type
        openai: openai's baseline
        simple: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
        """
        self.dps_type = 'simple'  # simple(stable) | openai(not available)
        self.case_num = 1  # fixed
        self.TEST = False  # False: train, True: test
        self.initialize_at_every_G_Ep = False

        # initialize directories for results
        self.dirname = f'reward_func_case{self.case_num}'
        self.dirfullname = '../results/{}/{}'.format('test' if self.TEST else 'train', self.dirname)
        self.initialize_result_dir(self.dirfullname)

        # fixed attributes
        self.OrcFx_model = OrcFxAPI.Model('DDPG_based_DPS.dat')
        self.DP_FORCE_windff = dp_force_windff.DP_FORCE()
        self.case_name_ind = self.case_num - 1

        self.DURATION_STARTING_TIME = - self.OrcFx_model['General'].StageDuration[0]
        self.DURATION_END_TIME = self.OrcFx_model['General'].StageDuration[1]
        self.SimulationTimeStep = self.OrcFx_model['General'].InnerTimeStep \
            if self.OrcFx_model['General'].InnerTimeStep else self.OrcFx_model['General'].ImplicitConstantTimeStep

        self.ModelSavePeriod = 1

        self.Gain_Recorder = True  # if self.TEST else False
        self.DPThrustHist_Recorder = True  # if self.TEST else False
        self.ship_pos_Recorder = True  # if self.TEST else False

        # nb_observations, nb_actions
        self.nb_observations = 30 + 3
        self.nb_actions = 9 + 3

        # filter_type
        self.filter_type = 'none'

        # (scaled) ZN gains (Kp_x, Kd_x, Ki_x, Kp_y, Kd_y, Ki_y, mp, md, mi)
        """
        # actual ZN-gains used in this paper:
        {'Kp_x': 3000000, 'Kd_x': 792375000, 'Ki_x': 300,
         'Kp_y': 6000000, 'Kd_y': 869250000, 'Ki_y': 1000,
         'mp': 18000000000, 'md': 2902500000000, 'mi': 2790700}
        """
        self.ZN_gains = {'Kp_x': 3000, 'Kd_x': 792375, 'Ki_x': 3,
                         'Kp_y': 6000, 'Kd_y': 869250, 'Ki_y': 10,
                         'mp': 18000000, 'md': 2902500000, 'mi': 27907}

        # set min, max ZN_gains limits
        self.min_actions, self.max_actions = {}, {}
        for gain_name in self.ZN_gains.keys():
            if 'i' in gain_name:
                self.min_actions[gain_name] = 0
                self.max_actions[gain_name] = 100
            else:
                self.min_actions[gain_name] = 0
                self.max_actions[gain_name] = 1000

        # params for DP
        self.Thrust_max = 2000e3  # [N]
        self.n_dp = 6
        self.time_constant_thrust = 50  # [step]
        self.use_kf_thrust = False  # kalman-filter is implemented by ignoring the effect of 'Wave load (1st order)' in OrcaFlex

        # params for ddpg (1)
        self.batch_size = 64
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3
        self.gamma = 0.99
        self.tau = 0.001

        # params for ddpg (2)
        self.apply_noise = True
        self.scale_obs = True
        self.scale_obs_by_const = (False, 10)
        self.scale_reward = True
        self.use_bn = True  # bn: batch normalization
        self.hl_a = 128  # hidden layer size of the actor
        self.hl_c = 64  # # hidden layer size of the critic
        self.var = 0.1  # control exploration
        self.var_decay_rate = 1.0  # 1.0 means no decay
        self.min_var = 0.05
        self.memory_size = (3600 * 12) * int(1 / self.SimulationTimeStep)  # [step], size of the replay buffer
        self.train_begin_step = self.batch_size

        # params for mu, sigma for pos, vel
        self.n_lim = 1 * int(1 / self.SimulationTimeStep)  # [step]

        # params for printing training progress
        self.train_progress_memory_size = int(self.SimulationTimeStep * (1 / self.SimulationTimeStep))  # [step]

        # fixed_action for n [steps]
        self.fixed_a_period = 1 #5 * int(1/self.SimulationTimeStep)  # [steps]

        # ignore the first n seconds
        # TEST에서 n[s]을 쓰는 이유는 트레이닝된 모델이 target position에서 왔다갔다 하는 상태 (즉 integral_err가 어느정도 쌓인 상태)에 fit되있기 때문
        self.ignoring_first_n_timestep = 0 if not self.TEST else 3600 #7200 #10800  # [s]

        # initial a
        self.init_gain_multiplier = [self.max_actions["Kp_x"], self.max_actions["Kd_x"], self.max_actions["Ki_x"]] * 3
        self.init_gain = np.array(list(self.ZN_gains.values())) * self.init_gain_multiplier

        # starting buffer size
        self.starting_buffer_size = 1 #(3600 * 1)  # [s]

        # localErr_list length
        self.localErr_list_len = 10  # [s]
        self.localErr_list_sampling_rate = 1 * int(1/self.SimulationTimeStep)  # [step]

        # gain change planning (given 'G_Ep')
        self.gain_change_plan = {'G_Ep': [0, 1],
                                 'zn_gain_multiplier': [1, 10]}  # 1: use 'zn_gain' as specified

        # gate_err_accumulation multiplier (conventional PID보다 drift가 적게 일어나므로, 그만큼을 보상해주기 위해)
        self.gate_err_accumulation_multiplier = 1  # '1' means ignoring this

        # peak_drifting_prevention
        # peaky_drifting_force를 잡을수 있도록 multiplying factor을 설정하여 준다.
        self.peak_drifting_prevention_factor = 1  # '1' means ignoring this

        # use max P, D gains (init_ZN_gain * max_actions)
        # = not using adaptive PD gains
        self.use_max_PD_gains = False

        # use I gain update (init_ZN_gain * max_actions)
        # = not using adaptive I gains
        self.use_max_I_gain = False

        # use 'gate for error accumulation' (output from the actor)
        self.use_gate_err_accumulation = True

        # clipping for obs, reward
        self.clip_range_obs = (-5, 5)  #(-10, 10)   # 너무 작으면, 배가 어디있는지 알수가없어서 수렴이 잘 할 수있도록 PD게인 컨트롤이 힘듦.
        self.clip_range_reward = (0, 2)


        # ================================================================================
        # set attributes
        self.start_time = time.time()
        self.sess = 0
        self.saver = 0
        self.G_Ep = 0
        self.ckpt_TEST = tf.train.get_checkpoint_state(f'../results/train/{self.dirname}/models') if self.TEST else None
        self.agent = ''

        # interactive attributes
        self.PositionX, self.PositionY, self.heading = 0, 0, 0
        self.VelocityX, self.VelocityY, self.AngularVelocityZ = 0, 0, 0
        self.accumulated_local_err = {'X': 0., 'Y': 0., 'R3': 0.}
        self.mu = {'X': 0., 'Y': 0., 'R3': 0., 'velX': 0., 'velY': 0., 'velR3': 0., 'T_MEAN': 0., 'n': 1,
                   'n_lim': self.n_lim}
        self.sigma = {'X': 0., 'Y': 0., 'R3': 0., 'velX': 0., 'velY': 0., 'velR3': 0., 'T_MEAN': 0.}
        self.T_MEAN = deque([0., 0.], maxlen=2)
        self.gain = []
        self.T_MEAN_storage = deque([0.], maxlen=self.n_lim)
        self.window_size = 1
        self.vel_dir_x, self.vel_dir_y, self.vel_dir_r3 = 0., 0., 0.
        self.env_dir_x, self.env_dir_y, self.env_dir_r3 = 0., 0., 0.
        self.localErr_list = deque(maxlen=self.localErr_list_len * int(1/self.SimulationTimeStep))
        self.gate_err_accumulation = [1, 1, 1]  # init val

        # interactive attributes - training progress
        self.mean_r = 0.
        self.mean_q = 0.
        self.loss_actor = 0.
        self.loss_critic = 0.
        self.param_noise_distance = 0.
        self.duration = 0.
        self.epoch_rewards = deque(maxlen=self.train_progress_memory_size)
        self.epoch_qs = deque(maxlen=self.train_progress_memory_size)
        self.epoch_actor_losses = deque(maxlen=self.train_progress_memory_size)
        self.epoch_critic_losses = deque(maxlen=self.train_progress_memory_size)
        self.epoch_adaptive_distances = deque(maxlen=self.train_progress_memory_size)
        self.a_count = 0
        self.a = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.del_a = [0]*self.nb_actions
        self.timestep = -99.

        # interactive attributes - ddpg_simple
        self.train_began_ddpg_simple = False
        self.ddpg_simple_scaler = DdpgSimpleScaler(self.nb_observations)

    def initialize_result_dir(self, dirfullname):
        try:
            os.mkdir(dirfullname)
        except FileExistsError:
            for fname in os.listdir(dirfullname):
                try:
                    os.unlink(dirfullname + '/' + fname)
                except PermissionError:
                    pass
            shutil.rmtree(dirfullname + '/models')
        os.mkdir(dirfullname + '/models')

    def initialize_mu_obs_sigma_obs(self, agent, do):
        if do:
            # get df with latest_fname_obs_rms
            path_obs_rms = '../results/train/' + self.dirname + '/models/'
            fnames_obs_rms = os.listdir(path_obs_rms)
            for fname in fnames_obs_rms:
                fnames_obs_rms.remove(fname) if 'obs_rms' not in fname else None
            latest_fname_obs_rms = sorted(fnames_obs_rms)[-1]
            print(f'# mu_obs, sigma_obs are initialized according to {latest_fname_obs_rms}')
            df_obs_rms = pd.read_csv(path_obs_rms + latest_fname_obs_rms).iloc[:, 1:]

            # set mu_obs, sigma_obs
            agent.ddpg_simple_scaler.mu_obs = df_obs_rms['obs_rms_mean']
            agent.ddpg_simple_scaler.sigma_obs = df_obs_rms['obs_rms_std']

    def get_agent(self, sess):
        if self.dps_type == 'openai':
            pass

        elif self.dps_type == 'simple':
            if self.use_bn:
                self.agent = DDPGSimple_bn_only(sess, self.nb_observations, self.nb_actions, self.actor_lr,
                                                self.critic_lr, self.tau, self.gamma, self.batch_size, self.memory_size,
                                                self.scale_obs, self.scale_obs_by_const, self.scale_reward, self.ddpg_simple_scaler, self.var,
                                                self.hl_a, self.hl_c, self.use_bn)
            else:
                self.agent = DDPGSimple(sess, self.nb_observations, self.nb_actions, self.actor_lr, self.critic_lr,
                                        self.tau, self.gamma, self.batch_size, self.memory_size,
                                        self.scale_obs, self.scale_obs_by_const, self.scale_reward, self.ddpg_simple_scaler,
                                        self.hl_a, self.hl_c, self.var)
            self.initialize_mu_obs_sigma_obs(self.agent, do=True if self.TEST else False)

    def initialize_sess_saver_agent(self):
        print("### initialize_sess_saver_agent ###")

        # set attributes
        tf.reset_default_graph()

        self.sess = U.single_threaded_session()

        self.get_agent(self.sess)
        self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=1000)

        self.sess.graph.finalize()
