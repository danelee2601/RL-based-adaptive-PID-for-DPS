import os, sys, re

sys.path.append(re.search(r'.*?\\DP', os.path.dirname(__file__)).group(0))  # register the root dir

import OrcFxAPI
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(1)

from baselines.ddpg.noise import *

from DP_DDPG.settings import DPSSettings
from DP_DDPG.DynamicPositioningSystem_helper import DPSHelper
from DP_DDPG.thruster_data import ThrusterData
from DP_DDPG.buffer_data import BufferData
from DP_DDPG.reward_func import reward_func
from DP_DDPG.KF import KFThrust

# class instances
dps_settings = DPSSettings()
dps_helper = DPSHelper(dps_settings)
kfs_thrust = {f'no.{i+1}': {key: KFThrust(dps_settings) for key in ['Fx', 'Fy', 'Mz']} for i in range(dps_settings.n_dp)} if dps_settings.use_kf_thrust else None


class Thruster(object):

    def GetParameter(self, params, paramName, default=None):
        if paramName in params:
            param = params[paramName]
            if isinstance(default, float):
                param = float(param)
            elif isinstance(default, float):
                param = float(param)
        elif default is not None:
            param = default
        else:
            raise Exception(
                'Parameter %s is required but is not included in the object parameters.' %
                paramName)
        return param

    def Initialise(self, info):
        self.periodNow = OrcFxAPI.Period(OrcFxAPI.pnInstantaneousValue)
        self.WorkspaceKey = info.ModelObject.Name + 'Thruster'
        self.WorkspaceKey2 = info.ModelObject.Name + 'Buffer'

        # initialize sess, saver, agent
        dps_settings.initialize_sess_saver_agent() if dps_settings.G_Ep == 0 else None

        # class instances
        thrusterData = ThrusterData()
        bufferData = BufferData(dps_settings.sess, dps_settings.saver, dps_settings,
                                load_model=True if dps_settings.TEST else False)

        # And store the external function parameters in thrusterdata:
        params = info.ObjectParameters

        # set the initial data from OrcFx
        thrusterData.TargetX = float(self.GetParameter(params, 'TargetX', 0))
        thrusterData.TargetY = float(self.GetParameter(params, 'TargetY', 0))
        thrusterData.TargetHeading = float(self.GetParameter(params, 'TargetHeading', 0)) * (np.pi / 180)  # [rad]

        thrusterData.ForceX = float(self.GetParameter(params, 'ForceX', 0))
        thrusterData.ForceY = float(self.GetParameter(params, 'ForceY', 0))
        thrusterData.MomentZ = float(self.GetParameter(params, 'MomentZ', 0))

        # dpt (from dp_force.cpp)
        thrusterData.dpt_zero = float(self.GetParameter(params, 'dpt_zero', 0))
        thrusterData.dpt_one = float(self.GetParameter(params, 'dpt_one', 0))
        thrusterData.dpt_two = float(self.GetParameter(params, 'dpt_two', 0))
        thrusterData.dpt_three = float(self.GetParameter(params, 'dpt_three', 0))
        thrusterData.dpt_four = float(self.GetParameter(params, 'dpt_four', 0))
        thrusterData.dpt_five = float(self.GetParameter(params, 'dpt_five', 0))
        thrusterData.dpt_six = float(self.GetParameter(params, 'dpt_six', 0))
        thrusterData.dpt_seven = float(self.GetParameter(params, 'dpt_seven', 0))
        thrusterData.dpt_eight = float(self.GetParameter(params, 'dpt_eight', 0))

        # w (from dp_force.cpp)
        thrusterData.w_zero_zero = float(self.GetParameter(params, 'w_zero_zero', 0))
        thrusterData.w_zero_one = float(self.GetParameter(params, 'w_zero_one', 0))
        thrusterData.w_zero_two = float(self.GetParameter(params, 'w_zero_two', 0))
        thrusterData.w_zero_three = float(self.GetParameter(params, 'w_zero_three', 0))
        thrusterData.w_zero_four = float(self.GetParameter(params, 'w_zero_four', 0))
        thrusterData.w_zero_five = float(self.GetParameter(params, 'w_zero_five', 0))
        thrusterData.w_zero_six = float(self.GetParameter(params, 'w_zero_six', 0))

        thrusterData.w_one_zero = float(self.GetParameter(params, 'w_one_zero', 0))
        thrusterData.w_one_one = float(self.GetParameter(params, 'w_one_one', 0))
        thrusterData.w_one_two = float(self.GetParameter(params, 'w_one_two', 0))
        thrusterData.w_one_three = float(self.GetParameter(params, 'w_one_three', 0))
        thrusterData.w_one_four = float(self.GetParameter(params, 'w_one_four', 0))
        thrusterData.w_one_five = float(self.GetParameter(params, 'w_one_five', 0))
        thrusterData.w_one_six = float(self.GetParameter(params, 'w_one_six', 0))

        thrusterData.w_two_zero = float(self.GetParameter(params, 'w_two_zero', 0))
        thrusterData.w_two_one = float(self.GetParameter(params, 'w_two_one', 0))
        thrusterData.w_two_two = float(self.GetParameter(params, 'w_two_two', 0))
        thrusterData.w_two_three = float(self.GetParameter(params, 'w_two_three', 0))
        thrusterData.w_two_four = float(self.GetParameter(params, 'w_two_four', 0))
        thrusterData.w_two_five = float(self.GetParameter(params, 'w_two_five', 0))
        thrusterData.w_two_six = float(self.GetParameter(params, 'w_two_six', 0))

        thrusterData.w_three_zero = float(self.GetParameter(params, 'w_three_zero', 0))
        thrusterData.w_three_one = float(self.GetParameter(params, 'w_three_one', 0))
        thrusterData.w_three_two = float(self.GetParameter(params, 'w_three_two', 0))
        thrusterData.w_three_three = float(self.GetParameter(params, 'w_three_three', 0))
        thrusterData.w_three_four = float(self.GetParameter(params, 'w_three_four', 0))
        thrusterData.w_three_five = float(self.GetParameter(params, 'w_three_five', 0))
        thrusterData.w_three_six = float(self.GetParameter(params, 'w_three_six', 0))

        thrusterData.w_four_zero = float(self.GetParameter(params, 'w_four_zero', 0))
        thrusterData.w_four_one = float(self.GetParameter(params, 'w_four_one', 0))
        thrusterData.w_four_two = float(self.GetParameter(params, 'w_four_two', 0))
        thrusterData.w_four_three = float(self.GetParameter(params, 'w_four_three', 0))
        thrusterData.w_four_four = float(self.GetParameter(params, 'w_four_four', 0))
        thrusterData.w_four_five = float(self.GetParameter(params, 'w_four_five', 0))
        thrusterData.w_four_six = float(self.GetParameter(params, 'w_four_six', 0))

        thrusterData.w_five_zero = float(self.GetParameter(params, 'w_five_zero', 0))
        thrusterData.w_five_one = float(self.GetParameter(params, 'w_five_one', 0))
        thrusterData.w_five_two = float(self.GetParameter(params, 'w_five_two', 0))
        thrusterData.w_five_three = float(self.GetParameter(params, 'w_five_three', 0))
        thrusterData.w_five_four = float(self.GetParameter(params, 'w_five_four', 0))
        thrusterData.w_five_five = float(self.GetParameter(params, 'w_five_five', 0))
        thrusterData.w_five_six = float(self.GetParameter(params, 'w_five_six', 0))

        thrusterData.w_six_zero = float(self.GetParameter(params, 'w_six_zero', 0))
        thrusterData.w_six_one = float(self.GetParameter(params, 'w_six_one', 0))
        thrusterData.w_six_two = float(self.GetParameter(params, 'w_six_two', 0))
        thrusterData.w_six_three = float(self.GetParameter(params, 'w_six_three', 0))
        thrusterData.w_six_four = float(self.GetParameter(params, 'w_six_four', 0))
        thrusterData.w_six_five = float(self.GetParameter(params, 'w_six_five', 0))
        thrusterData.w_six_six = float(self.GetParameter(params, 'w_six_six', 0))

        thrusterData.w_seven_zero = float(self.GetParameter(params, 'w_seven_zero', 0))
        thrusterData.w_seven_one = float(self.GetParameter(params, 'w_seven_one', 0))
        thrusterData.w_seven_two = float(self.GetParameter(params, 'w_seven_two', 0))
        thrusterData.w_seven_three = float(self.GetParameter(params, 'w_seven_three', 0))
        thrusterData.w_seven_four = float(self.GetParameter(params, 'w_seven_four', 0))
        thrusterData.w_seven_five = float(self.GetParameter(params, 'w_seven_five', 0))
        thrusterData.w_seven_six = float(self.GetParameter(params, 'w_seven_six', 0))

        # Put all of dpt, w together respectively
        self.dpt = np.zeros(shape=(9), dtype=np.int32)
        self.w = np.zeros(shape=(8, 7), dtype=np.float32)

        self.dpt_name_collection = [thrusterData.dpt_zero, thrusterData.dpt_one, thrusterData.dpt_two,
                                    thrusterData.dpt_three, thrusterData.dpt_four, thrusterData.dpt_five,
                                    thrusterData.dpt_six, thrusterData.dpt_seven, thrusterData.dpt_eight]
        self.w_name_collection = [thrusterData.w_zero_zero, thrusterData.w_zero_one, thrusterData.w_zero_two,
                                  thrusterData.w_zero_three, thrusterData.w_zero_four, thrusterData.w_zero_five,
                                  thrusterData.w_zero_six, thrusterData.w_one_zero, thrusterData.w_one_one,
                                  thrusterData.w_one_two, thrusterData.w_one_three, thrusterData.w_one_four,
                                  thrusterData.w_one_five, thrusterData.w_one_six, thrusterData.w_two_zero,
                                  thrusterData.w_two_one, thrusterData.w_two_two, thrusterData.w_two_three,
                                  thrusterData.w_two_four, thrusterData.w_two_five, thrusterData.w_two_six,
                                  thrusterData.w_three_zero, thrusterData.w_three_one, thrusterData.w_three_two,
                                  thrusterData.w_three_three, thrusterData.w_three_four, thrusterData.w_three_five,
                                  thrusterData.w_three_six, thrusterData.w_four_zero, thrusterData.w_four_one,
                                  thrusterData.w_four_two, thrusterData.w_four_three, thrusterData.w_four_four,
                                  thrusterData.w_four_five, thrusterData.w_four_six, thrusterData.w_five_zero,
                                  thrusterData.w_five_one, thrusterData.w_five_two, thrusterData.w_five_three,
                                  thrusterData.w_five_four, thrusterData.w_five_five, thrusterData.w_five_six,
                                  thrusterData.w_six_zero, thrusterData.w_six_one, thrusterData.w_six_two,
                                  thrusterData.w_six_three, thrusterData.w_six_four, thrusterData.w_six_five,
                                  thrusterData.w_six_six, thrusterData.w_seven_zero, thrusterData.w_seven_one,
                                  thrusterData.w_seven_two, thrusterData.w_seven_three, thrusterData.w_seven_four,
                                  thrusterData.w_seven_five, thrusterData.w_seven_six]

        for i in range(self.dpt.shape[0]):
            self.dpt[i] = self.dpt_name_collection[i]

        count = 0
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                self.w[i][j] = self.w_name_collection[count]
                count += 1

        # And save thrusterData in the Model Workspace dictionary:
        info.Workspace[self.WorkspaceKey] = thrusterData
        info.Workspace[self.WorkspaceKey2] = bufferData

    def Calculate(self, info):
        thrusterData, bufferData = info.Workspace[self.WorkspaceKey], info.Workspace[self.WorkspaceKey2]
        dataName, timestep = info.DataName, info.SimulationTime
        dps_settings.timestep = timestep

        if dataName.startswith('GlobalAppliedForceX'):

            # initialize
            if dps_settings.initialize_at_every_G_Ep and (round(timestep, 2) == dps_settings.DURATION_STARTING_TIME):
                dps_settings.accumulated_local_err['X'] = 0.
                dps_settings.accumulated_local_err['Y'] = 0.
                dps_settings.accumulated_local_err["R3"] = 0.

                dps_settings.mean_r = 0.
                dps_settings.mean_q = 0.
                dps_settings.loss_actor = 0.
                dps_settings.loss_critic = 0.
                dps_settings.param_noise_distance = 0.

            # update PositionX, PositionY, heading, VelocityX, VelocityY, AngularVelocityZ
            dps_helper.update_pos_vel(info, self.periodNow)

            # get "u" for "dp_force_windff.cp37-win_amd64.pyd"
            u = np.zeros(shape=(16,), dtype=np.float32)  # (obtained from OrcaFlex)
            u = dps_helper.update_u(u,
                                    dps_settings.PositionX, dps_settings.PositionY, dps_settings.heading,
                                    dps_settings.VelocityX, dps_settings.VelocityY, dps_settings.AngularVelocityZ,
                                    False, None, None, thrusterData)

            # update accumulated_err (for Ki)
            dps_helper.update_accumulated_local_err(dps_settings.accumulated_local_err,
                                                    dps_settings.PositionX, dps_settings.PositionY,
                                                    dps_settings.heading,
                                                    thrusterData)

            # update mu, sigma for pos, vel
            dps_helper.update_mu_sigma(dps_settings.mu, dps_settings.sigma,
                                       dps_settings.PositionX, dps_settings.PositionY, dps_settings.heading,
                                       dps_settings.VelocityX, dps_settings.VelocityY, dps_settings.AngularVelocityZ,
                                       dps_settings.T_MEAN[-1])

            # Initialize "ew" ( final output of dp_force.cpp )
            """
            ew[0][0]:F_x1 , ew[1][0]:F_y1 , ew[2][0]:M_z1  ( 1:1st DP )
            ew[0][1]:F_x2 , ew[1][1]:F_y2 , ew[2][1]:M_z2  ( 2:2nd DP )
            """
            ew = np.zeros(shape=(3, 8), dtype=np.float32)

            # get obs
            target_pos = np.array([thrusterData.TargetX, thrusterData.TargetY, thrusterData.TargetHeading])
            current_pos = np.array([dps_settings.PositionX, dps_settings.PositionY, dps_settings.heading])
            globalErr = target_pos - current_pos
            Rotation = np.array([[np.cos(current_pos[2]), -np.sin(current_pos[2]), 0],
                                 [np.sin(current_pos[2]), np.cos(current_pos[2]), 0],
                                 [0, 0, 1]])
            localErr = np.dot(Rotation, globalErr)

            localErr[2] *= (180 / np.pi)  # change rad to deg for heading (just in case I use 'constant_scaler')
            dps_settings.localErr_list.append(localErr)

            sampling_idices = np.arange(-1, -dps_settings.localErr_list.maxlen - 1, -dps_settings.localErr_list_sampling_rate)
            prev_localErrs = [dps_settings.localErr_list[i] for i in sampling_idices] if len(dps_settings.localErr_list) == dps_settings.localErr_list.maxlen else [([0]*len(localErr)) * dps_settings.localErr_list_len]
            prev_localErrs = np.ravel(prev_localErrs)

            obs = np.copy(prev_localErrs)
            obs = np.clip(obs, dps_settings.clip_range_obs[0], dps_settings.clip_range_obs[1])  # obs clipping
            abs_accumulated_local_err = np.sqrt( np.abs(list(dps_settings.accumulated_local_err.values())) + 1 )
            obs = np.concatenate(( obs, abs_accumulated_local_err ))  # add the integral term

            bufferData.s_list.append(obs)

            # Predict next actions |  pi : policy's symbol
            if ((dps_settings.a_count == dps_settings.fixed_a_period) or (len(bufferData.a_list) < 1)) and (dps_settings.ignoring_first_n_timestep <= timestep or dps_settings.G_Ep != 0):
                if dps_settings.dps_type == 'openai':
                    pass
                elif dps_settings.dps_type == 'simple':
                    dps_settings.a, q = dps_settings.agent.choose_action(obs, apply_noise=True if (dps_settings.TEST == False) and (dps_settings.apply_noise == True) else False)

                # seperate gain(a), gate_err_accumulation(a)
                dps_settings.a = dps_settings.a[:-3].copy()
                dps_settings.gate_err_accumulation = dps_settings.a[-3:].copy() if dps_settings.use_gate_err_accumulation else np.array([1, 1, 1])

                # set P, D gains to ZN gain and leave I_gain only
                if dps_settings.use_max_PD_gains:
                    dps_settings.a[0:2] = [1, 1]
                    dps_settings.a[3:5] = [1, 1]
                    dps_settings.a[6:8] = [1, 1]

                # set
                if dps_settings.use_max_I_gain:
                    dps_settings.a[2] = 1
                    dps_settings.a[5] = 1
                    dps_settings.a[8] = 1

                # I gain adjustment
                # env_force로인해 drifted 되는 구간에서는 max(I gain)을 쓴다. (rebound 되는 구간에서는 adaptive I gain을 쓴다.)
                local_x_dir, local_y_dir, local_r3_dir = - localErr
                if np.sign(-local_x_dir) == np.sign(dps_settings.accumulated_local_err['X']):
                    dps_settings.a[2] = 1
                if np.sign(-local_y_dir) == np.sign(dps_settings.accumulated_local_err['Y']):
                    dps_settings.a[5] = 1
                if np.sign(-local_r3_dir) == np.sign(dps_settings.accumulated_local_err['R3']):
                    dps_settings.a[8] = 1

                # update 'vel_dir', 'env_dir'
                dps_settings.vel_dir_x, dps_settings.vel_dir_y, dps_settings.vel_dir_r3 = -np.sign(dps_settings.localErr_list[-1] - dps_settings.localErr_list[-dps_settings.localErr_list_sampling_rate]) if len(dps_settings.localErr_list) > dps_settings.localErr_list_sampling_rate else [0., 0., 0.]
                dps_settings.env_dir_x, dps_settings.env_dir_y, dps_settings.env_dir_r3 = np.sign(-dps_settings.accumulated_local_err['X']), np.sign(-dps_settings.accumulated_local_err['Y']), np.sign(-dps_settings.accumulated_local_err['R3'])

                # store action, q in lists
                bufferData.a_list.append(np.concatenate((dps_settings.a, dps_settings.gate_err_accumulation)))
                dps_settings.epoch_qs.append(q)

                dps_settings.a_count = 0
                dps_settings.a_count += 1
            else:
                dps_settings.a_count += 1

            # get 'gains'
            if (dps_settings.ignoring_first_n_timestep <= timestep) or (dps_settings.G_Ep != 0):
                rescaled_a = dps_helper.action_rescaler(dps_settings.a, dps_settings.min_actions, dps_settings.max_actions)  # rescale a
                Kp_x, Kd_x, Ki_x, Kp_y, Kd_y, Ki_y, mp, md, mi = dps_helper.get_ddpg_gains(rescaled_a).values()  # get gains

                #dps_settings.init_gain = np.array([Kp_x, Kd_x, Ki_x,
                #                                   Kp_y, Kd_y, Ki_y,
                #                                   mp, md, mi])
            else:
                Kp_x, Kd_x, Ki_x, Kp_y, Kd_y, Ki_y, mp, md, mi = dps_settings.init_gain
                rescaled_a = dps_settings.init_gain

            if dps_settings.filter_type == 'kalman':
                "kalman-filter is implemented by ignoring the effect of 'Wave load (1st order)' in OrcaFlex"
                pass

            # update ew
            sp = np.array([thrusterData.TargetX, thrusterData.TargetY, thrusterData.TargetHeading], dtype=np.float32)
            dps_settings.DP_FORCE_windff.dp_force_windff(u, self.w, self.dpt, ew,
                                                         Kp_x, Kp_y, Kd_x, Kd_y, Ki_x, Ki_y, mp, md, mi, sp,
                                                         0., 0., 0.,
                                                         dps_settings.accumulated_local_err['X'],
                                                         dps_settings.accumulated_local_err['Y'],
                                                         dps_settings.accumulated_local_err['R3'],
                                                         )
            # separate ew and get thrusts
            bufferData.thrusts = dps_helper.get_thrusts(ew, bufferData.thrusts, kfs_thrust)
            Fx, Fy, Mz, T_SUM, t_mean = dps_helper.proc_thrusts(bufferData.thrusts)
            dps_settings.T_MEAN.append(t_mean)

            # Post-process ew | ew.shape: (3,8)
            globalPos = dps_helper.get_globalPos()
            rotation = dps_helper.rotation_func(globalPos)
            inv_rotation = np.linalg.inv(rotation)
            final_ew = np.dot(inv_rotation, np.array([[Fx], [Fy], [Mz]]))  # local to global

            # update forces
            thrusterData.ForceX = final_ew[0][0]
            thrusterData.ForceY = final_ew[1][0]
            thrusterData.MomentZ = final_ew[2][0]

            # get reward
            r = reward_func(dps_settings, thrusterData)
            dps_settings.epoch_rewards.append(r)

            # store (s,a,r,s',d) and train DDPG
            if not dps_settings.TEST and (dps_settings.ignoring_first_n_timestep <= timestep or dps_settings.G_Ep != 0):
                if len(bufferData.s_list) == 2 and len(bufferData.a_list) == 2:
                    new_obs, done = obs, False

                    if dps_settings.dps_type == 'openai':
                        pass

                    elif dps_settings.dps_type == 'simple':
                        dps_settings.agent.store_transition(bufferData.s_list[0], bufferData.a_list[0], r, new_obs,
                                                            update_mu_sigma=True if not dps_settings.TEST else False)
                        if (dps_settings.agent.pointer > dps_settings.train_begin_step) and (dps_settings.agent.len_buffer >= dps_settings.starting_buffer_size * (1/dps_settings.SimulationTimeStep)):
                            dps_helper.decay_var()
                            dps_helper.train_actor_critic(dps_settings, bufferData)

            # Print the current state every n seconds.
            if int(timestep) == bufferData.t_print:
                dps_helper.print_current_ship_DPstate(timestep, r,
                                                      Kp_x, Kd_x, Ki_x, Kp_y, Kd_y, Ki_y, mp, md, mi,
                                                      thrusterData, dps_settings.T_MEAN[-1], dps_settings.mu,
                                                      dps_settings.sigma, rescaled_a,
                                                      dps_settings.vel_dir_x, dps_settings.vel_dir_y, dps_settings.vel_dir_r3,
                                                      dps_settings.env_dir_x, dps_settings.env_dir_y, dps_settings.env_dir_r3)

            # record gains' state
            if dps_settings.Gain_Recorder:
                dps_helper.record_gains(timestep, Kp_x, Kd_x, Ki_x, Kp_y, Kd_y, Ki_y, mp, md, mi, dps_settings.gate_err_accumulation)

            # record (filtered) data
            if dps_settings.ship_pos_Recorder:
                dps_helper.record_ship_pos(timestep, dps_settings, dps_settings.filter_type)

            # record dp thrusts
            if dps_settings.DPThrustHist_Recorder:
                dps_helper.record_dp_thrust(timestep,
                                            bufferData.thrusts['no.1']['F'], bufferData.thrusts['no.2']['F'],
                                            bufferData.thrusts['no.3']['F'], bufferData.thrusts['no.4']['F'],
                                            bufferData.thrusts['no.5']['F'], bufferData.thrusts['no.6']['F'],
                                            T_SUM, dps_settings.T_MEAN[-1])

            # return the requested load component:
            info.Value = thrusterData.ForceX

        else:
            if dataName.startswith('GlobalAppliedForceY'):
                info.Value = thrusterData.ForceY

            if dataName.startswith('GlobalAppliedMomentZ'):
                info.Value = thrusterData.MomentZ

                # record training progress
                dps_helper.update_training_progress_params(bufferData, dps_settings.start_time)
                dps_helper.record_training_hist(dps_settings.G_Ep,
                                                timestep,
                                                dps_settings.mean_r,
                                                dps_settings.mean_q,
                                                dps_settings.loss_actor,
                                                dps_settings.loss_critic,
                                                dps_settings.param_noise_distance,
                                                dps_settings.duration)

                # print training progress
                if int(timestep) == bufferData.t_print:
                    bufferData.t_print += 10
                    dps_helper.print_training_progress()

                if int(timestep) == dps_settings.DURATION_END_TIME:
                    # save the model
                    if (dps_settings.G_Ep % dps_settings.ModelSavePeriod == 0) and not dps_settings.TEST:
                        dps_helper.save_model(dps_settings.saver, dps_settings.sess)

                    # save the values for obs_normalizing
                    if (dps_settings.scale_obs) and not dps_settings.TEST:
                        dps_helper.save_obs_mean_std(dps_settings.sess)

                    # increment G_Ep by 1
                    if not dps_settings.initialize_at_every_G_Ep:
                        dps_settings.G_Ep += 1
