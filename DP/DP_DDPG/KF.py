import numpy as np


class KFThrust(object):
    def __init__(self, dps_settings):
        """
        ref: https://blog.naver.com/danelee2601/221632608426
        F(=A) [nxn]: system matrix that releates the state at k-1 to the state at step k
        B [nx1]: it relates the control input(u_k) to the state(x_k)

        H [mxn]: matrix that relates the state to the measurement


        Q: process noise covariance. It represents the uncertainty in the process or model
        R: measurement noise covariance. It represents the unvertainty in the measurement

        P: error covariance.
        x0: initial state
        """

        dt = dps_settings.SimulationTimeStep
        self.F = np.array([[1, dt],
                           [0, 1]])
        self.H = np.array([1, 0]).reshape(1, -1)
        self.n = self.F.shape[1]
        self.m = self.H.shape[1]

        self.B = 0
        q = 0.1
        self.Q = np.array([[q, 0.0],
                           [0.0, q]])
        self.R = np.array([10]).reshape(1, 1)
        self.P = np.eye(self.n)
        self.x = np.zeros((self.n, 1))

    def predict(self, u=0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P),
                        (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
