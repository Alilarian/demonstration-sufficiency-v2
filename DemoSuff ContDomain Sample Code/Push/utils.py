import numpy as np
from scipy.interpolate import interp1d

def trajectory_optimization(env, trajectories, reward_param):
    """
    Simulating the planning phase in the trajectory space.

    This function takes a list of possible trajectories and a reward parameter, calculates the reward for each trajectory
    using `env.reward()`, and returns a sorted list of tuples containing (reward, trajectory) sorted by reward
    in descending order.

    Parameters
    ----------
    env : Env1
        An instance of the custom environment class Env1, providing a `reward()` function.
    trajectories : list of np.ndarray
        A list of trajectories, where each trajectory is represented as a NumPy array of features.
    reward_param : np.ndarray
        The parameter(s) used to calculate the reward for each trajectory.

    Returns
    -------
    list of tuple
        A list of tuples where each tuple contains:
          - reward (float): The calculated reward for the trajectory under the given reward_param.
          - traj_feature (np.ndarray): The corresponding trajectory features.
        The list is sorted by the reward in descending order.
    """
    traj_index_with_reward = []  # List of tuples (reward, trajectory_feature)
    for traj_index, traj_feature in enumerate(trajectories):
        curr_reward = env.reward(traj_feature, reward_param)
        traj_index_with_reward.append((curr_reward, traj_feature))

    # Sort by reward in descending order
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