import numpy as np
from math import sqrt
from math import atan2
from tools import Jacobian

class KalmanFilter:
    def __init__(self, x_in, P_in, F_in, H_in, R_in, Q_in):
        self.x = x_in
        self.P = P_in
        self.F = F_in
        self.H = H_in
        self.R = R_in
        self.Q = Q_in

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        # Calculate new estimates
        self.x = self.x + np.dot(K, z - np.dot(self.H, self.x))
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

    def update_ekf(self, z):
        H_j = Jacobian(self.x)

        S = np.dot(np.dot(H_j, self.P), H_j.T) + self.R

        K = np.dot(np.dot(self.P, H_j.T), np.linalg.inv(S))

        rad = sqrt(self.x[0] ** 2 + self.x[1] ** 2)
        rad_v = ((self.x[0] * self.x[2]) + (self.x[1] * self.x[3])) / rad
        hx = np.array([rad, atan2(self.x[1], self.x[0]), rad_v])

        y = z - hx
        y[1] = y[1] % -np.pi if y[1] < 0 else y[1] % np.pi

        self.x += np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, H_j), self.P)
