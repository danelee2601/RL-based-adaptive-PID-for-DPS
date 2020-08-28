import numpy as np
from pykalman import KalmanFilter


class ProcInput(object):

    def __init__(self, dps_settings):
        # given attributes - class instance
        self.dps_settings = dps_settings

        # set attributes
        self.inv_simultaion_time_step = int(1 / self.dps_settings.SimulationTimeStep)

        # init. attributes
        self.arr_pos_hist = []
        self.time_history, self.X_hist, self.Y_hist, self.R3_hist = [], [], [], []
        self.velX_hist, self.velY_hist, self.velR3_hist = [], [], []
        self.mean_X, self.mean_Y, self.mean_R3 = 0., 0., 0.
        self.std_X, self.std_Y, self.std_R3 = 0., 0., 0.

        self.sampled_X_hist, self.sampled_Y_hist, self.sampled_R3_hist = np.array([]), np.array([]), np.array([])
        self.smoo_X_hist, self.smoo_Y_hist, self.smoo_R3_hist = np.array([]), np.array([]), np.array([])
        self.smoo_velX_hist, self.smoo_velY_hist, self.smoo_velR3_hist = np.array([]), np.array([]), np.array([])

        # smoo
        self.smooX1, self.smooY1, self.smooR3_1 = 0., 0., 0.
        self.smoo_velX, self.smoo_velY, self.smoo_velR3 = 0., 0., 0.

        # smoo_err, smoo_vel_err
        self.smoo_errX1, self.smoo_errX2, self.smoo_errX3 = 0., 0., 0.
        self.smoo_errY1, self.smoo_errY2, self.smoo_errY3 = 0., 0., 0.
        self.smoo_errR3_1, self.smoo_errR3_2, self.smoo_errR3_3 = 0., 0., 0.
        self.smoo_vel_errX, self.smoo_vel_errY, self.smoo_vel_errR3 = 0., 0., 0.

    def update_arr_pos_hist(self, pos_hist):
        self.arr_pos_hist = np.array(pos_hist)

    def proc_arr_pos_hist(self):
        """
        process arr_post_hist and update
        self.time_history, self.X_hist, self.Y_hist, self.R3_hist
        """
        self.time_history, self.X_hist, self.Y_hist, self.R3_hist = self.arr_pos_hist[:, 0], \
                                                                    self.arr_pos_hist[:, 1], \
                                                                    self.arr_pos_hist[:, 2], \
                                                                    self.arr_pos_hist[:, 3],
        self.velX_hist, self.velY_hist, self.velR3_hist = self.arr_pos_hist[:, 4], \
                                                          self.arr_pos_hist[:, 5], \
                                                          self.arr_pos_hist[:, 6]

    def update_means(self):
        """update self.mean_X, self.mean_Y, self.mean_R3"""
        self.mean_X = np.mean(self.X_hist[- self.dps_settings.t_span_mean_std * self.inv_simultaion_time_step:])
        self.mean_Y = np.mean(self.Y_hist[- self.dps_settings.t_span_mean_std * self.inv_simultaion_time_step:])
        self.mean_R3 = np.mean(self.R3_hist[- self.dps_settings.t_span_mean_std * self.inv_simultaion_time_step:])

    def update_stds(self):
        """update self.std_X, self.std_Y, self.std_R3"""
        self.std_X = np.std(self.X_hist[- self.dps_settings.t_span_mean_std * self.inv_simultaion_time_step:])
        self.std_Y = np.std(self.Y_hist[- self.dps_settings.t_span_mean_std * self.inv_simultaion_time_step:])
        self.std_R3 = np.std(self.R3_hist[- self.dps_settings.t_span_mean_std * self.inv_simultaion_time_step:])

    def get_sampled_hist(self, hist):
        return hist[::-self.dps_settings.sampling_period * self.inv_simultaion_time_step][::-1]

    def update_sampled_hists(self):
        """update self.sampled_X_hist, self.sampled_Y_hist, self.sampled_R3_hist"""
        self.sampled_X_hist = self.get_sampled_hist(self.X_hist)
        self.sampled_Y_hist = self.get_sampled_hist(self.Y_hist)
        self.sampled_R3_hist = self.get_sampled_hist(self.R3_hist)

    def update_smoo_hists(self):
        """
        update self.smoo_X_hist, self.smoo_Y_hist, self.smoo_R3_hist according to filter_type
        """
        if self.dps_settings.filter_type == 'none':
            self.smoo_X_hist = self.X_hist
            self.smoo_Y_hist = self.Y_hist
            self.smoo_R3_hist = self.R3_hist
            self.smoo_velX_hist = self.velX_hist
            self.smoo_velY_hist = self.velY_hist
            self.smoo_velR3_hist = self.velR3_hist

        elif self.dps_settings.filter_type == 'kalman':
            self.update_sampled_hists()
            kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1, )
            self.smoo_X_hist, _ = kf.filter(self.sampled_X_hist)
            self.smoo_Y_hist, _ = kf.filter(self.sampled_Y_hist)
            self.smoo_R3_hist, _ = kf.filter(self.sampled_R3_hist)

    def update_smoo_XYR3(self):
        self.smooX1 = self.smoo_X_hist[-1]
        self.smooY1 = self.smoo_Y_hist[-1]
        self.smooR3_1 = self.smoo_R3_hist[-1]

    def update_smoo_errXYR3(self, target_x, target_y, target_heading):
        """
        update
        self.smoo_errX3, self.smoo_errX2, self.smoo_errX1
        self.smoo_errY3, self.smoo_errY2, self.smoo_errY1
        self.smoo_errR3_3, self.smoo_errR3_2, self.smoo_errR3_1
        """
        self.smoo_errX3, self.smoo_errX2, self.smoo_errX1 = self.smoo_X_hist[-3:] - target_x
        self.smoo_errY3, self.smoo_errY2, self.smoo_errY1 = self.smoo_Y_hist[-3:] - target_y
        self.smoo_errR3_3, self.smoo_errR3_2, self.smoo_errR3_1 = self.smoo_R3_hist[-3:] - target_heading

    def update_smoo_vel_errXYR3(self):
        """
        update
        self.smoo_vel_errX, self.smoo_vel_errY, self.smoo_vel_errR3
        with self.smoo_errX[\d]{1}, self.smoo_errY[\d]{1}, self.smoo_errR3_[\d]{1}
        """
        self.smoo_vel_errX = (self.smoo_errX1 - self.smoo_errX2) / self.dps_settings.sampling_period
        self.smoo_vel_errY = (self.smoo_errY1 - self.smoo_errY2) / self.dps_settings.sampling_period
        self.smoo_vel_errR3 = (self.smoo_errR3_1 - self.smoo_errR3_2) / self.dps_settings.sampling_period


class SimplifyVel(object):
    """
    it has methods to simplify the given input.
    It makes directionality of vel, acc same towards both sides.
    """

    def __init__(self, sampling_period):
        # given attributes - class instance
        self.sampling_period = sampling_period

        # interactive attributes
        self.err_x, self.err_y, self.err_r3, self.err_vel_x, self.err_vel_y, self.err_vel_r3 = 0., 0., 0., 0., 0., 0.
        self.sim_vel_errX, self.sim_vel_errY, self.sim_vel_errR3 = 0., 0., 0.

    def update_err_pos_vel(self, proc_input, dps_settings):
        if dps_settings.filter_type == 'none':
            self.err_x, self.err_y, self.err_r3 = dps_settings.PositionX, dps_settings.PositionY, dps_settings.heading
            self.err_r3, self.err_vel_x, self.err_vel_y, = \
                dps_settings.VelocityX, dps_settings.VelocityY, dps_settings.AngularVelocityZ

        elif dps_settings.filter_type == 'kalman':
            self.err_x, self.err_y, self.err_r3 = proc_input.smoo_errX1, proc_input.smoo_errY1, proc_input.smoo_errR3_1
            self.err_r3, self.err_vel_x, self.err_vel_y, = \
                proc_input.smoo_vel_errX, proc_input.smoo_vel_errY, proc_input.smoo_vel_errR3

    def simplify(self, err, err_vel):
        """it does the main operation"""
        sim_smoo_vel_err_x = - err_vel if err < 0 else err_vel
        return sim_smoo_vel_err_x

    def update_sim_vel(self):
        self.sim_vel_errX = self.simplify(self.err_x, self.err_vel_x)
        self.sim_vel_errY = self.simplify(self.err_y, self.err_vel_y)
        self.sim_vel_errR3 = self.simplify(self.err_r3, self.sim_vel_errR3)
