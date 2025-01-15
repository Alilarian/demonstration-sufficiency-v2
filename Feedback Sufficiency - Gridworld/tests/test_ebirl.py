import sys
import os
import yaml
import numpy as np

# Get current and parent directory to handle import paths
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from env import gridworld_env
from agent.q_learning_agent import ValueIteration
from reward_learning.ebirl import EBIRL
from data_generation.generate_data import GridWorldMDPDataGenerator

# Function to load YAML configuration
def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Function to run the agent movement test using the configuration
def test_birl(config):
    # Extract configuration parameters
    render_mode = config['env_config']['render_mode']
    size = config['env_config']['size']
    noise_prob = config['env_config']['noise_prob']
    seed = config['env_config']['seed']
    gamma = config['env_config']['gamma']
    epsilon = float(config['algorithm_config']['epsilon'])

    num_steps = config['bayesian_irl_config']['num_steps']
    step_stdev = config['bayesian_irl_config']['step_stdev']
    beta = config['bayesian_irl_config']['beta']
    normalize = config['bayesian_irl_config']['normalize']
    adaptive = config['bayesian_irl_config']['adaptive']

    # Initialize the environment with the loaded configuration
    env = gridworld_env.NoisyLinearRewardFeaturizedGridWorldEnv(
        gamma=gamma, render_mode=render_mode, size=size, noise_prob=noise_prob
    )
    env.reset(seed=seed)

    # Step 1: Compute the optimal policy for the true environment
    val_iter = ValueIteration(mdp=env)
    state_values = val_iter.run_value_iteration(epsilon=epsilon)
    true_policy = val_iter.get_optimal_policy()
    #preferences = GridWorldMDPDataGenerator(env).generate_pairwise_comparisons(strategy="same_start_state", num_trajs=6)
    estops = GridWorldMDPDataGenerator(env).generate_estop(beta=beta, num_trajs=10)
    
    # Step 2: Run BIRL to learn the MAP reward
    birl = EBIRL(env, demos=estops, beta=beta)
    birl.run_mcmc(num_steps, step_stdev, normalize=normalize, adaptive=adaptive)
    map_reward = birl.get_map_solution()
    print("Learned MAP Reward:", map_reward)

if __name__ == "__main__":
    # Load parameters from the YAML configuration file
    config = load_yaml_config("configs/gridworld_config.yaml")
    
    # Run the test using the loaded configuration
    test_birl(config)