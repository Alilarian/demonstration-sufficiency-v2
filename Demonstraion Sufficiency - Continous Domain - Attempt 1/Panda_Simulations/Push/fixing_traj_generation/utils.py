import numpy as np
from scipy.interpolate import interp1d


def quaternion_angular_difference(q1, q2):

    dot_product = np.dot(q1, q2)
    angle = 2 * np.arccos(np.abs(dot_product))
    return angle

class Trajectory():

    def __init__(self, xi, T):
        self.T = T
        self.n, self.m = np.shape(xi)
        self.traj = []
        xi = np.asarray(xi)
        timesteps = np.linspace(0, self.T, self.n)
        for idx in range(self.m):
            self.traj.append(interp1d(timesteps, xi[:, idx], kind='linear'))
        
    def get_waypoint(self, t):
        if t < 0.0:
            t = 0.0
        if t > self.T:
            t = self.T
        waypoint = np.array([0.] * self.m)
        for idx in range(self.m):
            waypoint[idx] = self.traj[idx](t)
        return waypoint