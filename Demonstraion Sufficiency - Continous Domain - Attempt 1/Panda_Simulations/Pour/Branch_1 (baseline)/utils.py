import numpy as np
from scipy.interpolate import interp1d

def trajectory_optimization(env, trajectories, traj_feats, reward_param):
    """
    Simulating the planning phase in the trajectory space.

    This function takes a list of the all possible traj_feats and a reward parameter, calculates the reward for each trajectory,
    and returns a sorted list of tuples containing the reward and the trajectory index, sorted by reward in descending order.

    Parameters:
    traj_feats (list of np.ndarray): A list of traj_feats, where each trajectory is represented as a NumPy array of features.
    reward_param (any): The parameter(s) used to calculate the reward for each trajectory.

    Returns:
    list of tuple: A list of tuples where each tuple contains:
                   - curr_reward (float): The calculated reward for the trajectory.
                   - traj_index (int): The index of the trajectory in the original list.
                   The list is sorted by the reward in descending order.
    """
    traj_index_with_reward = []  # List of tuples (reward, trajectory index)
    #for traj_index, traj_feature in enumerate(traj_feats):
    for traj, traj_feature in zip(trajectories, traj_feats):
        curr_reward = env.reward(traj_feature, reward_param)
        #traj_index_with_reward.append((curr_reward, traj_index))
        traj_index_with_reward.append((curr_reward, traj_feature, traj))

    return sorted(traj_index_with_reward, key=lambda x: x[0], reverse=True)

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