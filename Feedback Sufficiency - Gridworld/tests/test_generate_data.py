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

from envs import gridworld_env
from agent.q_learning_agent import ValueIteration, PolicyEvaluation
from data_generation.generate_data import GridWorldMDPDataGenerator



# Function to load YAML configuration
def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_yaml_config("configs/gridworld_config.yaml")
# Extract configuration parameters
render_mode = config['env_config']['render_mode']
size = config['env_config']['size']
noise_prob = config['env_config']['noise_prob']
seed = config['env_config']['seed']
gamma = config['env_config']['gamma']
steps = config['env_config']['steps']
#sleep_time = config['env_config']['sleep_time']
epsilon = float(config['algorithm_config']['epsilon'])

# Initialize the environment with the loaded configuration
env = gridworld_env.NoisyLinearRewardFeaturizedGridWorldEnv(
    gamma=gamma, render_mode=render_mode, size=size, noise_prob=noise_prob
)

print(env.feature_weights)

env.reset(seed=seed)

#envs.set_feature_weights(np.random.rand(4))

val_iter = ValueIteration(mdp=env)
state_values = val_iter.run_value_iteration(epsilon=epsilon)
policy = val_iter.get_optimal_policy()
qvalues = val_iter.get_q_values()

datagen = GridWorldMDPDataGenerator(env=env, q_values=qvalues)

#print(datagen.generate_optimal_demo(num_trajs=1, start_states=[0]))
#print(datagen.generate_random_demo(num_trajs=1, start_states=[0]))
#print(len(datagen.generate_pairwise_comparisons(strategy="same_start_state", num_trajs=10)))

#estops = datagen.generate_estop(beta=1, num_trajs=10)

#for i in estops:
#    traj, t = i
#    print(t)
#    print(traj)

comparisons = datagen.generate_pairwise_comparisons(strategy="same_start_state", num_trajs=1)

for i in comparisons:
    print(i)
#    print(i)
#    print()