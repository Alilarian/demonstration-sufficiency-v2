import numpy as np
from env_1 import Env1
from algos import *
import math
from scipy.stats import norm


start_position = np.array([0.4, 0.0, 0.25])
goal_position = np.array([0.75, 0.0, 0.1])
xi0 = np.linspace(start_position, goal_position, 3)

true_reward_param = np.array([ 0.90181928,  0.38649398, -0.19324699])

# Define the list of alpha values
num_generated_demos = 50000

env = Env1(visualize=False)

demonstration_space, xi_star, trajectory_space, f_star, rewards = human_demo_2(env, xi0, true_reward_param,\
                                                                           n_samples=num_generated_demos, n_demos=5000)


np.save('demonstration_space.npy', demonstration_space)
np.save('xi_star.npy', xi_star)
np.save('trajectory_space.npy', trajectory_space)
np.save('f_star.npy', f_star)
np.save('rewards.npy', rewards)