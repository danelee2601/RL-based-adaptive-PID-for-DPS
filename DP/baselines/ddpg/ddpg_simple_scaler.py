import numpy as np


class DdpgSimpleScaler(object):

    def __init__(self, nb_observations):
        self.mu_obs = np.zeros(nb_observations)
        self.sigma_obs = np.ones(nb_observations)

        self.mu_r = 0.
        self.sigma_r = 1.

        self.epsilon = 1e-10
        self.min_n = 3

    def update_mu_sigma_obs(self, n, obs):
        n = self.corrected_n(n)
        self.mu_obs = (1 / n) * obs + (n - 1) / n * self.mu_obs
        self.sigma_obs = np.sqrt((n - 2) / (n - 1 + self.epsilon) * self.sigma_obs ** 2 + 1 / n * (obs - self.mu_obs) ** 2)

    def update_mu_sigma_r(self, n, r):
        n = self.corrected_n(n)
        self.mu_r = (1 / n) * r + (n - 1) / n * self.mu_r
        self.sigma_r = np.sqrt((n - 2) / (n - 1 + self.epsilon) * self.sigma_r ** 2 + 1 / n * (r - self.mu_r) ** 2)

    def corrected_n(self, n):
        return self.min_n if n <= self.min_n else n
