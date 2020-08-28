import os, sys, re, glob
sys.path.append(re.search(r'.*?\\DP', os.path.dirname(__file__)).group(0))  # register the root dir

import OrcFxAPI
import dp_force_windff  # included in Lib (.pyd is in DP/dp_force_source/rev03)
import numpy as np


class DPSSettings(object):

    def __init__(self):
        # initialize directories for results
        self.dirfullname = '../results/ZN/temp/'
        self.initialize_at_every_G_Ep = False

        self.initialize_result_dir()

        # set attributes
        self.OrcFx_model = OrcFxAPI.Model('ZN_based_DPS.dat')
        self.DP_FORCE_windff = dp_force_windff.DP_FORCE()

        self.DURATION_STARTING_TIME = - self.OrcFx_model['General'].StageDuration[0]
        self.DURATION_END_TIME = self.OrcFx_model['General'].StageDuration[1]
        self.SimulationTimeStep = self.OrcFx_model['General'].InnerTimeStep \
            if self.OrcFx_model['General'].InnerTimeStep else self.OrcFx_model['General'].ImplicitConstantTimeStep

        self.Gain_Recorder = True
        self.DPThrustHist_Recorder = True
        self.ship_pos_Recorder = True

        # wind feed-forward
        self.use_wind_feedforward = False

        # motion feed-forward
        self.use_pred_model = False

        # filter_type
        self.filter_type = 'none'

        # Kalman filter option
        # ...

        # parameters for DP
        self.Thrust_max = 2000e3  # [N]
        self.n_dp = 6
        self.time_constant_thrust = 50  # [step]

        # interactive attributes
        self.PositionX, self.PositionY, self.heading = 0, 0, 0
        self.VelocityX, self.VelocityY, self.AngularVelocityZ = 0, 0, 0
        self.G_Ep = 0
        self.accumulated_local_err = {'X': 0., 'Y': 0., 'R3': 0.}
        self.t_print = 0

        # not used but exists because of compatibility
        self.gate_err_accumulation = [1, 1, 1]  # init val
        self.mu = {'X': 0, 'Y': 0, 'R3': 0}
        self.sigma = {'X': 0, 'Y': 0, 'R3': 0}
        self.gate_err_accumulation_multiplier = 1
        self.timestep = -99.

    def initialize_result_dir(self):
        try:
            os.makedirs(self.dirfullname)
        except FileExistsError:
            for fname in [f for f in os.listdir(self.dirfullname) if '.' in f]:
                os.unlink(self.dirfullname + '/' + fname)
