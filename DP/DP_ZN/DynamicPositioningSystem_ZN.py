import os, sys, re

sys.path.append(re.search(r'.*?\\DP', os.path.dirname(__file__)).group(0))  # register the root dir

import OrcFxAPI
import numpy as np

from DP_ZN.settings import DPSSettings
from DP_ZN.DynamicPositioningSystem_helper import DPSHelper
from DP_ZN.thruster_data import ThrusterData
from DP_ZN.pred_model import PredModel, PredModelMemory

# class instances
dps_settings = DPSSettings()
dps_helper = DPSHelper(dps_settings)
pred_model = PredModel(dps_settings)  # not related to this paper, please ignore this
pred_model_memory = PredModelMemory(dps_settings, pred_model)  # not related to this paper, please ignore this


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

        # class instances
        thrusterData = ThrusterData()

        # initialize the one_step_pred_model
        if dps_settings.G_Ep == 0:
            pred_model.update_model()

        # And store the external function parameters in thrusterdata:
        params = info.ObjectParameters

        # set the initial data from OrcFx
        thrusterData.TargetX = float(self.GetParameter(params, 'TargetX', 0))
        thrusterData.TargetY = float(self.GetParameter(params, 'TargetY', 0))
        thrusterData.TargetHeading = float(self.GetParameter(params, 'TargetHeading', 0)) * (np.pi / 180)  # [rad]

        thrusterData.Kp_x = float(self.GetParameter(params, 'Kp_x', 0))
        thrusterData.Kd_x = float(self.GetParameter(params, 'Kd_x', 0))
        thrusterData.Ki_x = float(self.GetParameter(params, 'Ki_x', 0))
        thrusterData.Kp_y = float(self.GetParameter(params, 'Kp_y', 0))
        thrusterData.Kd_y = float(self.GetParameter(params, 'Kd_y', 0))
        thrusterData.Ki_y = float(self.GetParameter(params, 'Ki_y', 0))
        thrusterData.mp = float(self.GetParameter(params, 'mp', 0))
        thrusterData.md = float(self.GetParameter(params, 'md', 0))
        thrusterData.mi = float(self.GetParameter(params, 'mi', 0))

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

        # interactive attributes
        self.thrusts = {f'no.{no}': {'Fx': 0., 'Fy': 0., 'F': 0., 'Mz': 0.}
                        for no in range(1, dps_settings.n_dp + 1)}
        dps_settings.t_print = 0

    def Calculate(self, info):
        thrusterData = info.Workspace[self.WorkspaceKey]

        dataName = info.DataName

        timestep = info.SimulationTime
        dps_settings.timestep = timestep

        if dataName.startswith('GlobalAppliedForceX'):

            # initialize
            if dps_settings.initialize_at_every_G_Ep and (round(timestep, 2) == dps_settings.DURATION_STARTING_TIME):
                dps_settings.accumulated_local_err['X'] = 0.
                dps_settings.accumulated_local_err['Y'] = 0.
                dps_settings.accumulated_local_err["R3"] = 0.

            # update PositionX, PositionY, heading, VelocityX, VelocityY, AngularVelocityZ
            dps_helper.update_pos_vel(info, self.periodNow)

            # get "u" for "dp_force.cpp"
            u = np.zeros(shape=(16,), dtype=np.float32)  # (obtained from OrcaFlex)
            u = dps_helper.update_u(u,
                                    dps_settings.PositionX, dps_settings.PositionY, dps_settings.heading,
                                    dps_settings.VelocityX, dps_settings.VelocityY, dps_settings.AngularVelocityZ,
                                    dps_settings.use_pred_model, pred_model_memory, pred_model, thrusterData,
                                    timestep)

            # update accumulated_err (for Ki)
            dps_helper.update_accumulated_local_err(dps_settings.accumulated_local_err,
                                                    dps_settings.PositionX, dps_settings.PositionY,
                                                    dps_settings.heading,
                                                    thrusterData, pred_model)

            # Initialize "ew" ( final output of dp_force.cpp )
            """
            ew[0][0]:F_x1 , ew[1][0]:F_y1 , ew[2][0]:M_z1  ( 1:1st DP )
            ew[0][1]:F_x2 , ew[1][1]:F_y2 , ew[2][1]:M_z2  ( 2:2nd DP )
            """
            ew = np.zeros(shape=(3, 8), dtype=np.float32)

            # get gains
            Kp_x, Kd_x, Ki_x, Kp_y, Kd_y, Ki_y, mp, md, mi = \
                thrusterData.Kp_x, thrusterData.Kd_x, thrusterData.Ki_x, \
                thrusterData.Kp_y, thrusterData.Kd_y, thrusterData.Ki_y, \
                thrusterData.mp, thrusterData.md, thrusterData.mi

            if dps_settings.filter_type == 'kalman':
                "kalman-filter is implemented by ignoring the effect of 'Wave load (1st order)' in OrcaFlex"
                pass

            # update ew
            sp = np.array([thrusterData.TargetX, thrusterData.TargetY, thrusterData.TargetHeading],
                          dtype=np.float32)

            local_wind_forces = \
                [info.ModelObject.TimeHistory(f'Wind {i}', self.periodNow)[0]
                 for i in ['Lx force', 'Ly force', 'Lz moment']] if dps_settings.use_wind_feedforward else [0., 0., 0.]

            dps_settings.DP_FORCE_windff.dp_force_windff(u, self.w, self.dpt, ew,
                                                         Kp_x, Kp_y, Kd_x, Kd_y, Ki_x, Ki_y, mp, md, mi, sp,
                                                         local_wind_forces[0], local_wind_forces[1],
                                                         local_wind_forces[2],
                                                         dps_settings.accumulated_local_err['X'],
                                                         dps_settings.accumulated_local_err['Y'],
                                                         dps_settings.accumulated_local_err['R3'])

            # seperate ew and get thrusts
            self.thrusts = dps_helper.get_thrusts(ew, self.thrusts)
            Fx, Fy, Mz, T_SUM, T_MEAN = dps_helper.proc_thrusts(self.thrusts)

            # Post-process ew | ew.shape: (3,8)
            globalPos = dps_helper.get_globalPos()
            rotation = dps_helper.rotation_func(globalPos)
            inv_rotation = np.linalg.inv(rotation)
            final_ew = np.dot(inv_rotation, np.array([[Fx], [Fy], [Mz]]))  # local to global

            # update forces
            thrusterData.ForceX = final_ew[0][0]
            thrusterData.ForceY = final_ew[1][0]
            thrusterData.MomentZ = final_ew[2][0]

            # Print the current state every n seconds.
            if int(timestep) == dps_settings.t_print:
                dps_helper.print_current_ship_DPstate(timestep)

            if int(timestep) == dps_settings.t_print:
                dps_settings.t_print += 10

            # record gains' state
            if dps_settings.Gain_Recorder:
                dps_helper.record_gains(timestep, Kp_x, Kd_x, Ki_x, Kp_y, Kd_y, Ki_y, mp, md, mi, dps_settings.gate_err_accumulation)

            # record (filtered) data
            if dps_settings.ship_pos_Recorder:
                dps_helper.record_ship_pos(timestep, dps_settings, dps_settings.filter_type)

            # record dp thrusts
            if dps_settings.DPThrustHist_Recorder:
                dps_helper.record_dp_thrust(timestep,
                                            self.thrusts['no.1']['F'], self.thrusts['no.2']['F'],
                                            self.thrusts['no.3']['F'], self.thrusts['no.4']['F'],
                                            self.thrusts['no.5']['F'], self.thrusts['no.6']['F'],
                                            T_SUM, T_MEAN)

            # return the requested load component:
            info.Value = thrusterData.ForceX

        else:
            if dataName.startswith('GlobalAppliedForceY'):
                info.Value = thrusterData.ForceY

            if dataName.startswith('GlobalAppliedMomentZ'):
                info.Value = thrusterData.MomentZ

                # increment G_Ep
                if not dps_settings.initialize_at_every_G_Ep:
                    dps_settings.G_Ep += 1
