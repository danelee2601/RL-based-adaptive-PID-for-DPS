import os
import time
from textwrap import dedent
import numpy as np
import pandas as pd


class DPSHelper(object):

    def __init__(self, dps_settings):

        # given class instances
        self.dps_settings = dps_settings

        # interactive attributes
        self.f_record_append_gains = ''
        self.f_record_dp_thrust = ''
        self.f_record_ship_pos = ''

    def check_if_existing(self, fname):
        answer = False
        if fname in os.listdir(self.dps_settings.dirfullname):
            answer = True
        return answer

    def record_gains(self, timestep,
                     Kp_x, Kd_x, Ki_x, Kp_y, Kd_y, Ki_y, mp, md, mi, gate_err_accumulation,
                     fname='gain_hist.csv'):
        """record gains in csv file"""
        gains = [timestep, Kp_x, Kd_x, Ki_x, Kp_y, Kd_y, Ki_y, mp, md, mi, *gate_err_accumulation]

        fullfname = self.dps_settings.dirfullname + '/{}'.format(fname)

        if not self.check_if_existing(fname):
            with open(fullfname, 'w') as f:
                f.write('timestep,Kp_x,Kd_x,Ki_x,Kp_y,Kd_y,Ki_y,mp,md,mi,gate_err_accumulationX,gate_err_accumulationY,gate_err_accumulationR3\n')
            # update the state of f
            self.f_record_append_gains = open(fullfname, 'a')
        else:
            if timestep == self.dps_settings.DURATION_STARTING_TIME:
                self.f_record_append_gains = open(fullfname, 'a')

            self.f_record_append_gains.write('{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(*gains, *gate_err_accumulation))

            if int(timestep) == self.dps_settings.DURATION_END_TIME:
                self.f_record_append_gains.close()

    def record_dp_thrust(self, *args, fname='thrust_hist.csv'):
        """
        receive thrust(s) and save in csv file
        1st col: timestep
        2nd ~ nth col: Thrust
        nth+1 col: T_SUM
        nth+2 col: T_MEAN
        """
        record_content = [arg for arg in args]
        timestep = record_content[0]
        n_thrusters = len(record_content) - 3
        colnames = ['T{}'.format(i) for i in range(1, n_thrusters + 1)]
        colnames.insert(0, 'timestep')
        colnames.append('T_SUM')
        colnames.append('T_MEAN')

        fullfname = self.dps_settings.dirfullname + '/{}'.format(fname)

        if not self.check_if_existing(fname):
            with open(fullfname, 'w') as f:
                [f.write('{},'.format(colname)) for colname in colnames]
                f.write('\n')
                # update the state of f
                self.f_record_dp_thrust = open(fullfname, 'a')
        else:
            if timestep == self.dps_settings.DURATION_STARTING_TIME:
                self.f_record_dp_thrust = open(fullfname, 'a')

            [self.f_record_dp_thrust.write('{},'.format(rc)) for rc in record_content]
            self.f_record_dp_thrust.write('\n')

            if int(timestep) == self.dps_settings.DURATION_END_TIME:
                self.f_record_dp_thrust.close()

    def record_ship_pos(self, timestep, dps_settings, filter_type):
        """
        save data filtered by kalman filter in csv file
        1st col: timestep
        2nd, 3rd, 4th col: x, y, r3
        5th, 6th, 7th col: vel_x, vel_y, vel_r3
        """
        #fname = 'ship_pos_filtered_by_{}.csv'.format(filter_type)
        fname = 'ship_pos.csv'
        data = [timestep,
                dps_settings.PositionX, dps_settings.PositionY, dps_settings.heading * (180/np.pi),
                dps_settings.VelocityX, dps_settings.VelocityY, dps_settings.AngularVelocityZ * (180/np.pi),
                dps_settings.mu['X'], dps_settings.mu['Y'], dps_settings.mu['R3'] * (180/np.pi),
                dps_settings.sigma['X'], dps_settings.sigma['Y'], dps_settings.sigma['R3'] * (180/np.pi),
                dps_settings.accumulated_local_err['X'], dps_settings.accumulated_local_err['Y'], dps_settings.accumulated_local_err['R3'],
                ]

        fullfname = self.dps_settings.dirfullname + '/{}'.format(fname)
        colnames = ['timestep', 'x', 'y', 'r3', 'vel_x', 'vel_y', 'vel_r3',
                    'muX', 'muY', 'muR3', 'sigmaX', 'sigmaY', 'sigmaR3',
                    'accumulated_local_errX', 'accumulated_local_errY', 'accumulated_local_errR3',
                    ]

        if not self.check_if_existing(fname):
            with open(fullfname, 'w') as f:
                f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(*colnames))
            # update the state of f
            self.f_record_ship_pos = open(fullfname, 'a')
        else:
            try:
                self.f_record_ship_pos.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(*data))
            except ValueError:
                self.f_record_ship_pos = open(fullfname, 'a')

            if int(timestep) == self.dps_settings.DURATION_END_TIME:
                self.f_record_ship_pos.close()

    def record_training_hist(self, G_Ep, timestep, mean_r, mean_q, loss_actor, loss_critic, param_noise_distance,
                             duration):
        data = [G_Ep, timestep, mean_r, mean_q, loss_actor, loss_critic, param_noise_distance, duration]
        colnames = ['Glo_Ep', 'timestep', 'mean_r', 'mean_q', 'loss_actor', 'loss_critic', 'param_noise_distance',
                    'duration']

        if (str(loss_actor) == 'nan') or (timestep < self.dps_settings.ignoring_first_n_timestep):
            data = [G_Ep, timestep, mean_r] + [0] * (len(colnames) - 3)

        fname = 'training_hist.csv'
        fullfname = self.dps_settings.dirfullname + '/{}'.format(fname)

        if not self.check_if_existing(fname):
            with open(fullfname, 'w') as f:
                [f.write('{},'.format(colname)) for colname in colnames]
                f.write('\n')
        with open(fullfname, 'a') as f:
            [f.write('{},'.format(d)) for d in data]
            f.write('\n')

    def get_globalPos(self):
        """
        globalPos will be used
        to convert the local position in the "sum_ew"
        to global given the "SP" in the global coordinate.
        """
        globalPos = np.zeros((3, 1))
        globalPos[0][0] = self.dps_settings.PositionX
        globalPos[1][0] = self.dps_settings.PositionY
        globalPos[2][0] = self.dps_settings.heading
        return globalPos

    def rotation_func(self, ang):
        """
        function needed to process 'ew'
        ew is the output from dp_force.pyd
        """
        rotation = np.array([[np.cos(ang[2][0]), np.sin(ang[2][0]), 0],
                             [-np.sin(ang[2][0]), np.cos(ang[2][0]), 0],
                             [0, 0, 1]])
        return rotation

    def action_rescaler(self, a, min_actions: dict, max_actions: dict):
        rescaled_action_list = []
        for a_element, min_action, max_action in zip(a, min_actions.values(), max_actions.values()):
            y = (max_action - min_action) / (1 - 0) * a_element + min_action
            rescaled_action_list.append(y)
        return rescaled_action_list

    def get_ddpg_gains(self, rescaled_a):
        """
        rescaled_a consists of multipliers for each gain.
        """
        ddpg_gains = {}
        idx = 0
        for gain_name, zn_gain in self.dps_settings.ZN_gains.items():
            ddpg_gains[gain_name] = zn_gain * rescaled_a[idx]
            idx += 1
        return ddpg_gains

    def correct_heading_ang(self, heading_ang):
        """
        correct the heading angle and convert it into radians
        correction e.g: 280 deg -> 10 deg
        """
        divided_heading = heading_ang / 360
        shifted_heading = np.modf(divided_heading)
        decimal_part_only = shifted_heading[0]
        redeclared_heading = decimal_part_only * 360
        heading_ang = redeclared_heading

        heading_ang = heading_ang * (np.pi / 180)  # to convert it into radians

        if heading_ang > np.pi:
            heading_ang = heading_ang - 2 * np.pi
        elif heading_ang <= -np.pi:
            heading_ang = heading_ang + 2 * np.pi
        return heading_ang

    def update_pos_vel(self, info, periodNow):
        """
        update
        PositionX, PositionY, heading,
        VelocityX, VelocityY, AngularVelocityZ
        """
        # pos
        self.dps_settings.PositionX, self.dps_settings.PositionY = info.InstantaneousCalculationData.Position[:2]
        self.dps_settings.heading = \
            self.correct_heading_ang(info.ModelObject.TimeHistory('Rotation 3', periodNow)[0])  # [rad]

        # vel
        self.dps_settings.VelocityX, self.dps_settings.VelocityY = info.InstantaneousCalculationData.Velocity[:2]
        self.dps_settings.AngularVelocityZ = info.InstantaneousCalculationData.AngularVelocity[2]  # [rad/s]

    def update_u(self, u, x, y, r3, vel_x, vel_y, vel_r3,
                 use_pred_model, pred_model_memory, pred_model,
                 thrusterData, ts=None):
        """
        update u with X, Y, heading, vel_X, vel_Y, vel_R3
        global pos -> u[1], u[2], u[3]
        global vel ->  u[4], u[5], u[6]

        note: global pos, vel will be converted into local coordinate system in the dp_force.cpp
        """
        u[1] = x
        u[2] = y
        u[3] = r3  # [rad]
        u[4] = vel_x
        u[5] = vel_y
        u[6] = vel_r3  # [rad] when imported from OrcaFlex

        if use_pred_model and \
                (pred_model_memory.len_memory >= pred_model.batch_size) and \
                (pred_model_memory.len_memory > pred_model.starting_memory_size):

            pred_model.active_status = True

            model_input = np.array(
                [pred_model_memory.pos_hist['X'][-pred_model.past_length:][::-pred_model.sampling_rate][::-1],
                 pred_model_memory.pos_hist['Y'][-pred_model.past_length:][::-pred_model.sampling_rate][::-1],
                 pred_model_memory.pos_hist['R3'][-pred_model.past_length:][::-pred_model.sampling_rate][::-1],
                 pred_model_memory.vel_hist['velX'][-pred_model.past_length:][::-pred_model.sampling_rate][::-1],
                 pred_model_memory.vel_hist['velY'][-pred_model.past_length:][::-pred_model.sampling_rate][::-1],
                 pred_model_memory.vel_hist['velR3'][-pred_model.past_length:][::-pred_model.sampling_rate][::-1],
                 pred_model_memory.acc_hist['accX'][-pred_model.past_length:][::-pred_model.sampling_rate][::-1],
                 pred_model_memory.acc_hist['accY'][-pred_model.past_length:][::-pred_model.sampling_rate][::-1],
                 pred_model_memory.acc_hist['accR3'][-pred_model.past_length:][::-pred_model.sampling_rate][::-1],
                 pred_model_memory.thrust_hist['Tx'][-pred_model.past_length:][::-pred_model.sampling_rate][::-1],
                 pred_model_memory.thrust_hist['Ty'][-pred_model.past_length:][::-pred_model.sampling_rate][::-1],
                 pred_model_memory.thrust_hist['Mz'][-pred_model.past_length:][::-pred_model.sampling_rate][::-1],
                 pred_model_memory.wind_hist['wfX'][-pred_model.past_length:][::-pred_model.sampling_rate][::-1],
                 pred_model_memory.wind_hist['wfY'][-pred_model.past_length:][::-pred_model.sampling_rate][::-1],
                 pred_model_memory.wind_hist['wfR3'][-pred_model.past_length:][::-pred_model.sampling_rate][::-1]
                 ])

            # data scaling
            if pred_model.scale_obs:
                sc_model_input = (model_input.T - pred_model_memory.mu_X) / pred_model_memory.sigma_X
            else:
                sc_model_input = model_input.T
            sc_model_input = np.expand_dims(sc_model_input, axis=0)

            # predict - pred_model
            sc_pred = pred_model.predict(sc_model_input)
            pred = sc_pred * pred_model_memory.sigma_Y + pred_model_memory.mu_Y

            # log pred
            pred_model.f_pred_hist.write(f'{ts},{pred[0]},{pred[1]},{pred[2] * (180/np.pi)}\n')

            # apply amplifier
            if pred_model.use_amplifier:
                pred[0] = pred_model.amplifier(pred[0])
                pred[1] = pred_model.amplifier(pred[1])
                pred[2] = pred_model.amplifier(pred[2])

            # store in pred_memory
            pred_model.pred_memory['x'].append(pred[0])
            pred_model.pred_memory['y'].append(pred[1])
            pred_model.pred_memory['r3'].append(pred[2])

            # update u matrix
            u[1] = pred_model.pred_memory['x'][-1]
            u[2] = pred_model.pred_memory['y'][-1]
            u[3] = pred_model.pred_memory['r3'][-1]

            if pred_model.use_pred_2nd_derivative_err:
                u[4] = (pred_model.pred_memory['x'][-1] - pred_model.pred_memory['x'][-2]) / self.dps_settings.SimulationTimeStep
                u[5] = (pred_model.pred_memory['y'][-1] - pred_model.pred_memory['y'][-2]) / self.dps_settings.SimulationTimeStep
                u[6] = (pred_model.pred_memory['r3'][-1] - pred_model.pred_memory['r3'][-2]) / self.dps_settings.SimulationTimeStep

        return u

    def train_actor_critic(self, dps_settings, bufferData):
        if dps_settings.dps_type == 'openai':
            for t_train in range(dps_settings.nb_train_steps):
                # Adapt param noise, if necessary.
                if (dps_settings.memory.nb_entries >= dps_settings.batch_size) and \
                        (t_train % dps_settings.param_noise_adaption_interval == 0):
                    """
                    the perturbed policy is sampled at the beginning of each episode a`nd kept fixed for the entire rollout.
                    """
                    distance = dps_settings.agent.adapt_param_noise()
                    dps_settings.epoch_adaptive_distances.append(distance)

                len_buffer = len(dps_settings.agent.memory.observations0)
                if len_buffer >= dps_settings.batch_size:
                    cl, al = dps_settings.agent.train()
                    dps_settings.epoch_actor_losses.append(al)
                    dps_settings.epoch_critic_losses.append(cl)

                    dps_settings.agent.update_target_net()

        elif dps_settings.dps_type == 'simple':
            loss_actor, loss_critic = dps_settings.agent.learn()
            dps_settings.epoch_actor_losses.append(loss_actor)
            dps_settings.epoch_critic_losses.append(loss_critic)
            self.dps_settings.train_began_ddpg_simple = True

    def time_lagged_thrust(self, pre_force, ordered_force: int):
        """
        give the time-lag effect to thrust
        ref: 'Hydrodynamic Forces and Maneuvering Characteristics of Ships at Low Advance Speed'
        """
        if self.dps_settings.time_constant_thrust < 1:
            raise ValueError('time_constant must be greater than 1.')
        force_change = (ordered_force - pre_force) / self.dps_settings.time_constant_thrust
        return pre_force + force_change

    def get_thrusts(self, ew, prev_thrusts, kfs_thrust: dict=None):
        """
        get thrusts from ew
        if kfs_thrust, kalman filter is applied else, no filter  (kfs: kalman filters)
        """
        thrusts = {}
        for i in range(self.dps_settings.n_dp):
            no = i + 1

            Fx = self.time_lagged_thrust(prev_thrusts[f'no.{no}']['Fx'], ew[0, i])
            Fy = self.time_lagged_thrust(prev_thrusts[f'no.{no}']['Fy'], ew[1, i])
            Mz = self.time_lagged_thrust(prev_thrusts[f'no.{no}']['Mz'], ew[2, i])

            if kfs_thrust:
                kfs_thrust[f'no.{no}']['Fx'].update(Fx)
                kfs_thrust[f'no.{no}']['Fy'].update(Fy)
                kfs_thrust[f'no.{no}']['Mz'].update(Mz)

                Fx = kfs_thrust[f'no.{no}']['Fx'].predict()[0][0]
                Fy = kfs_thrust[f'no.{no}']['Fy'].predict()[0][0]
                Mz = kfs_thrust[f'no.{no}']['Mz'].predict()[0][0]

            F = np.sqrt(Fx ** 2 + Fy ** 2)

            thrust = {'Fx': Fx, 'Fy': Fy, 'F': F, 'Mz': Mz}
            thrusts[f'no.{no}'] = thrust
        return thrusts

    def proc_thrusts(self, thrusts,):
        """process thrusts and get Fx, Fy, Mz, T_SUM, T_MEAN"""
        Fx = sum([val['Fx'] for val in thrusts.values()])
        Fy = sum([val['Fy'] for val in thrusts.values()])
        Mz = sum([val['Mz'] for val in thrusts.values()])
        T_SUM = sum([val['F'] for val in thrusts.values()])
        T_MEAN = np.mean([val['F'] for val in thrusts.values()])
        return Fx, Fy, Mz, T_SUM, T_MEAN

    def print_current_ship_DPstate(self, timestep, r,
                                   Kp_x, Kd_x, Ki_x, Kp_y, Kd_y, Ki_y, mp, md, mi,
                                   thrustData, T_MEAN,
                                   mu, sigma, rescaled_a,
                                   vel_dir_x, vel_dir_y, vel_dir_r3,
                                   env_dir_x, env_dir_y, env_dir_r3):

        if self.dps_settings.dps_type == 'openai':
            len_buffer = len(self.dps_settings.agent.memory.observations0)
        elif self.dps_settings.dps_type == 'simple':
            len_buffer = self.dps_settings.agent.len_buffer

        print("\n\n==================================================================")
        print("G_Ep: {}, timestep: {:0.1f}, Reward: {:0.3f}, len_buffer: {}".format(self.dps_settings.G_Ep,
                                                                                    timestep, r, len_buffer))
        print("PositionX: {:0.1f}, PositionY: {:0.1f}, heading: {:0.1f}[deg]".format(
            self.dps_settings.PositionX, self.dps_settings.PositionY, self.dps_settings.heading * (180 / np.pi)))
        print('mu_X:{:0.2f}, mu_Y:{:0.2f}, mu_R3:{:0.2f}, mu_velX:{:0.2f}, mu_velY:{:0.2f}, mu_velR3:{:0.2f}'.format(
            mu['X'], mu['Y'], mu['R3']*(180/np.pi), mu['velX'], mu['velY'], mu['velR3']*(180/np.pi)))
        print('sigma_X:{:0.2f}, sigma_Y:{:0.2f}, sigma_R3:{:0.2f}, sigma_velX:{:0.2f}, sigma_velY:{:0.2f}, sigma_velR3:{:0.2f}'.format(
            sigma['X'], sigma['Y'], sigma['R3']*(180/np.pi), sigma['velX'], sigma['velY'], sigma['velR3']*(180/np.pi)))
        print("[rescaled_a] Kp_x:{:0.2f}, Kd_x:{:0.2f}, Ki_x:{:0.2f}, "
              "Kp_y:{:0.2f}, Kd_y:{:0.2f}, Ki_y:{:0.2f}, "
              "mp:{:0.2f}, md:{:0.2f}, mi:{:0.2f}".format(*rescaled_a))
        print("Kp_x:{:0.0f}, Kd_x:{:0.0f}, Ki_x:{:0.0f}, "
              "Kp_y:{:0.0f}, Kd_y:{:0.0f}, Ki_y:{:0.0f}, "
              "mp:{:0.0f}, md:{:0.0f}, mi:{:0.0f}".format(
            Kp_x, Kd_x, Ki_x, Kp_y, Kd_y, Ki_y, mp, md, mi))
        print('ForceX: {:0.0f}, ForceY: {:0.0f}, MomentZ: {:0.0f}, T_MEAN: {:0.0f}'.format(
            thrustData.ForceX, thrustData.ForceY, thrustData.MomentZ, T_MEAN))
        print("accumulated_local_err['X']: {:0.2f}, accumulated_local_err['Y']: {:0.2f}, accumulated_local_err['R3']: {:0.2f}".format( *list(self.dps_settings.accumulated_local_err.values()) ))
        print("vel_dir_x: {}, vel_dir_y: {}, vel_dir_r3: {}".format(vel_dir_x, vel_dir_y, vel_dir_r3))
        print("env_dir_x: {}, env_dir_y: {}, env_dir_r3: {}".format(env_dir_x, env_dir_y, env_dir_r3))
        print("gate_err_accumulationX: {:0.1f}, gate_err_accumulationY: {:0.1f}, gate_err_accumulationR3: {:0.1f}".format(*self.dps_settings.gate_err_accumulation))

    def update_training_progress_params(self, bufferData, start_time):
        """
        update
        dps_settings.mean_r, dps_settings.mean_q
        dps_settings.loss_actor, dps_settings.loss_critic
        dps_settings.param_noise_distance
        dps_settings.duration
        """
        self.dps_settings.mean_r = np.mean(self.dps_settings.epoch_rewards)
        self.dps_settings.mean_q = np.mean(self.dps_settings.epoch_qs)
        self.dps_settings.loss_actor = np.mean(self.dps_settings.epoch_actor_losses)
        self.dps_settings.loss_critic = np.mean(self.dps_settings.epoch_critic_losses)
        self.dps_settings.param_noise_distance = np.mean(self.dps_settings.epoch_adaptive_distances)
        self.dps_settings.duration = time.time() - start_time

    def print_training_progress(self):
        print('================================')
        print(f'* Averaged over {self.dps_settings.train_progress_memory_size * self.dps_settings.SimulationTimeStep}s')
        print('mean_r: {:0.1f}'.format(self.dps_settings.mean_r))
        print('mean_q: {:0.3f}'.format(self.dps_settings.mean_q))
        print('loss_actor: {:0.5f}'.format(self.dps_settings.loss_actor))
        print('loss_critic: {:0.5f}'.format(self.dps_settings.loss_critic))
        print('param_noise_distance: {:0.2f}'.format(self.dps_settings.param_noise_distance))
        print('var: {:0.3f}'.format(self.dps_settings.var)) if self.dps_settings.dps_type == 'simple' else None
        print('duration(s): {:0.1f}'.format(self.dps_settings.duration))
        print('================================\n')

    def save_model(self, saver, sess):
        fname = f'/models/model_GEp_{self.dps_settings.G_Ep}.ckpt'
        saver.save(sess, self.dps_settings.dirfullname + fname)
        print(f"\n{fname} is saved.\n")

    def save_obs_mean_std(self, sess):
        fname = f'/models/model_GEp_{self.dps_settings.G_Ep}_obs_rms.csv'
        if self.dps_settings.dps_type == 'openai':
            obs_rms_mean, obs_rms_std = sess.run([self.dps_settings.agent.obs_rms.mean, self.dps_settings.agent.obs_rms.std])
            d = {'obs_rms_mean': obs_rms_mean, 'obs_rms_std': obs_rms_std}
            pd.DataFrame(d).to_csv(self.dps_settings.dirfullname + fname)
            print(f'{fname} is saved.\n')
        elif self.dps_settings.dps_type == 'simple':
            d = {'obs_rms_mean': self.dps_settings.agent.ddpg_simple_scaler.mu_obs,
                 'obs_rms_std': self.dps_settings.agent.ddpg_simple_scaler.sigma_obs}
            pd.DataFrame(d).to_csv(self.dps_settings.dirfullname + fname)
            print(f'{fname} is saved.\n')
            # rms: recorded mean, sigma

    def get_R(self, heading_angle):
        """R is used to convert the coordinate system"""
        R = np.array([[np.cos(heading_angle), -np.sin(heading_angle), 0],
                      [np.sin(heading_angle), np.cos(heading_angle), 0],
                      [0, 0, 1]])
        return R

    def lo_wf2glo_wf(self, lwf_x, lwf_y, lwf_r3, heading_angle):
        """convert the local wind force/moment to global wind force/moment"""
        inv_R = np.linalg.inv(self.get_R(heading_angle))
        xyr3 = np.array([lwf_x, lwf_y, lwf_r3]).reshape(-1, 1)
        gwf_x, gwf_y, gwf_r3 = np.dot(inv_R, xyr3).ravel()
        return gwf_x, gwf_y, gwf_r3

    def get_Wind_GX_force_moment(self, info, periodNow, heading):
        gwf_x, gwf_y, gwf_r3 = \
            self.lo_wf2glo_wf(info.ModelObject.TimeHistory('Wind Lx force', periodNow)[0],
                              info.ModelObject.TimeHistory('Wind Ly force', periodNow)[0],
                              info.ModelObject.TimeHistory('Wind Lz moment', periodNow)[0],
                              heading)
        return gwf_x, gwf_y, gwf_r3

    def update_accumulated_local_err(self, accumulated_local_err: dict, x, y, r3, thrusterData, pred_model=None):
        """
        x, y, r3 are global coordinates
        r3 is in radian
        """
        if pred_model and pred_model.use_pred_integral_err and pred_model.active_status:
            x = pred_model.pred_memory['x'][-1]
            y = pred_model.pred_memory['y'][-1]
            r3 = pred_model.pred_memory['r3'][-1]

        target_pos = np.array([thrusterData.TargetX, thrusterData.TargetY, thrusterData.TargetHeading])
        local_err = np.dot(self.get_R(r3), target_pos) - np.dot(self.get_R(r3), np.array([x, y, r3]).reshape(-1, 1)).ravel()

        accumulated_local_err['X'] += (local_err[0] * self.dps_settings.SimulationTimeStep) * (self.dps_settings.gate_err_accumulation[0] * self.dps_settings.gate_err_accumulation_multiplier)
        accumulated_local_err['Y'] += (local_err[1] * self.dps_settings.SimulationTimeStep) * (self.dps_settings.gate_err_accumulation[1] * self.dps_settings.gate_err_accumulation_multiplier)
        accumulated_local_err['R3'] += (local_err[2] * self.dps_settings.SimulationTimeStep) * (self.dps_settings.gate_err_accumulation[2] * self.dps_settings.gate_err_accumulation_multiplier)

    def update_mu_sigma(self, mu: dict, sigma: dict,
                        x: float, y: float, r3: float,
                        vel_x: float, vel_y: float, vel_r3: float,
                        t_mean: float):
        for i, j in zip(mu.keys(), [x, y, r3, vel_x, vel_y, vel_r3, t_mean]):
            mu[i] = ((mu['n'] - 1) / mu['n']) * mu[i] + (1 / mu['n']) * j
            sigma[i] = np.sqrt(
                ((mu['n'] - 2) / (mu['n'] - 1 + 1e-10)) * (sigma[i] ** 2) + 1 / mu['n'] * (j - mu[i]) ** 2)
        mu['n'] = mu['n'] + 1 if not mu['n_lim'] else min(mu['n'] + 1, mu['n_lim'])  # update n

    def decay_var(self):
        """decay the action randomness"""
        self.dps_settings.var *= self.dps_settings.var_decay_rate
        if self.dps_settings.var < self.dps_settings.min_var:
            self.dps_settings.var = self.dps_settings.min_var

    def get_diff_xyr3(self, bufferData):
        prev, current = 0, 1
        x, y, r3 = 0, 1, 2
        if len(bufferData.localErr_list) == 2:
            diff_x = (bufferData.localErr_list[current][x] - bufferData.localErr_list[prev][x])
            diff_y = (bufferData.localErr_list[current][y] - bufferData.localErr_list[prev][y])
            diff_r3 = (bufferData.localErr_list[current][r3] - bufferData.localErr_list[prev][r3]) * (180 / np.pi)
            if bufferData.localErr_list[current][x] >= 0:
                diff_x *= -1
            if bufferData.localErr_list[current][y] >= 0:
                diff_y *= -1
            if bufferData.localErr_list[current][r3] >= 0:
                diff_r3 *= -1

            diff_x = np.clip(diff_x, None, 0)
            diff_y = np.clip(diff_y, None, 0)
            diff_r3 = np.clip(diff_r3, None, 0)

        else:
            diff_x, diff_y, diff_r3 = 0., 0., 0.

        return diff_x, diff_y, diff_r3
