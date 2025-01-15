import sys
import os
import time
import yaml
import numpy as np
import random

import copy
import math
from scipy.stats import norm

# Get current and parent directory to handle import paths
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from agent.q_learning_agent import ValueIteration, PolicyEvaluation
from utils.common_helper import calculate_expected_value_difference
from env import gridworld_env


# Function to generate a random policy for each state
def generate_random_policy(env):
    policy = []
    for state in range(env.num_states):  # assuming discrete states
        rand_action = env.action_space.sample()  # assign a random action to each state
        policy.append((state, rand_action))
    return policy



with open("configs/gridworld_config.yaml", 'r') as file:
    config = yaml.safe_load(file)


# Extract configuration parameters
render_mode = config['env_config']['render_mode']
size = config['env_config']['size']
noise_prob = config['env_config']['noise_prob']
seed = config['env_config']['seed']
gamma = config['env_config']['gamma']
steps = config['env_config']['steps']
sleep_time = config['env_config']['sleep_time']
epsilon = float(config['algorithm_config']['epsilon'])
#print(gamma)


# Initialize the environment with the loaded configuration
env = gridworld_env.NoisyLinearRewardFeaturizedGridWorldEnv(
    gamma=gamma, render_mode=render_mode, size=size, noise_prob=noise_prob
)

# generate random policy for the gridworld
random_policy = generate_random_policy(env)

print(random_policy)

random_policy_val = PolicyEvaluation(env, random_policy, uniform_random=True).run_policy_evaluation(epsilon)

print("Random Policy Val: ", np.mean(random_policy_val))

env.reset(seed=seed)

val_iter = ValueIteration(mdp=env)
state_values = val_iter.run_value_iteration(epsilon=epsilon)
policy = val_iter.get_optimal_policy()
qvalues = val_iter.get_q_values()

print("Optimal Policy Val: ", np.mean(state_values))
