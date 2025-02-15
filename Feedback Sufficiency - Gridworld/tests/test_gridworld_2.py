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

from envs import gridworld_env2

# Function to load YAML configuration
def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Function to run the agent movement test using the configuration
def test_agent_movement(env_config):
    # Extract configuration parameters
    render_mode = env_config['render_mode']
    size = env_config['size']
    noise_prob = env_config['noise_prob']
    seed = env_config['seed']
    steps = env_config['steps']
    sleep_time = env_config['sleep_time']
    gamma = env_config['gamma']

    color_to_feature_map = {
        "blue": [1, 0, 0],
        "red": [0, 1, 0],
        "black": [0, 0, 1]  # 'black' indicates a terminal state
    }

    custom_grid_features = [
        ["blue", "red", "blue"],
        ["blue", "blue", "black"]
    ]

    custom_feature_weights = [0.5, -0.5, 1]  # Example custom weights for features

    env = gridworld_env2.NoisyLinearRewardFeaturizedGridWorldEnv(
        gamma=0.99,
        color_to_feature_map=color_to_feature_map,
        grid_features=custom_grid_features,
        custom_feature_weights=custom_feature_weights,
        render_mode=render_mode
    )
    
    # Reset the environment to start a new episode with the given seed
    observation, info = env.reset(seed=seed)
    print("Initial observation:", observation)
    
    # Run the agent for the specified number of steps
    for step in range(steps):
        # Randomly choose an action (move in a direction)
        action = env.action_space.sample()
        
        # Step the environment with the chosen action
        observation, reward, terminated, _ = env.step(action)
        
        # Print the agent's new location after the step
        print(f"Step {step + 1}: Agent at {observation['agent']}, Reward: {reward}")
        
        # Pause for a short time (for rendering purposes)
        time.sleep(sleep_time)
        
        # If the agent reaches the goal, reset the environment
        if terminated:
            print("Agent reached the target!")
            observation = env.reset(seed=seed)

    # Close the environment window once the test is complete
    env.close()

# Main function
if __name__ == "__main__":
    # Load parameters from the YAML configuration file
    config = load_yaml_config("configs/gridworld_config.yaml")
    
    # Run the test using the loaded configuration
    test_agent_movement(config['env_config'])