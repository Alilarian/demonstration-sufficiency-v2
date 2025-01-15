import sys
import os
import time
import yaml
import numpy as np
import random


# Get current and parent directory to handle import paths
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from env import gridworld_env
from agent.q_learning_agent import ValueIteration, PolicyEvaluation

# Function to load YAML configuration
def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Function to run the agent movement test using the configuration
def test_value_iteration(config):
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

    env.reset(seed=seed)

    val_iter = ValueIteration(mdp=env)
    state_values = val_iter.run_value_iteration(epsilon=epsilon)
    policy = val_iter.get_optimal_policy()
    qvalues = val_iter.get_q_values()


    return state_values, policy, qvalues

if __name__ == "__main__":
    # Load parameters from the YAML configuration file
    config = load_yaml_config("configs/gridworld_config.yaml")
    
    # Run the test using the loaded configuration
    state_values, policy, qvalues = test_value_iteration(config)
     
    print(state_values)
    print(policy)
    print(qvalues)