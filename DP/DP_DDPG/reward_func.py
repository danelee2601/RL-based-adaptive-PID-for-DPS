import numpy as np


def get_R(heading_angle):
    """R is used to convert the coordinate system"""
    R = np.array([[np.cos(heading_angle), -np.sin(heading_angle), 0],
                  [np.sin(heading_angle), np.cos(heading_angle), 0],
                  [0, 0, 1]])
    return R


def reward_func(dps_settings, thrusterData):
    x, y, r3 = dps_settings.PositionX, dps_settings.PositionY, dps_settings.AngularVelocityZ
    target_pos = np.array([thrusterData.TargetX, thrusterData.TargetY, thrusterData.TargetHeading])
    local_err = np.dot(get_R(r3), target_pos) - np.dot(get_R(r3), np.array([x, y, r3]).reshape(-1, 1)).ravel()

    err_xy = np.sqrt(local_err[0] ** 2 + local_err[1] ** 2)

    # clipping
    err_xy = np.clip(err_xy, dps_settings.clip_range_reward[0], dps_settings.clip_range_reward[1])

    # define 'r'
    r = - err_xy + 1
    r *= 0.1

    return r
