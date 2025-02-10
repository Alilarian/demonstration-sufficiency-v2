import random
import copy
from scipy.stats import norm
import numpy as np
import argparse
import sys
import os
import yaml
import logging

# Get current and parent directory to handle import paths
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from env import gridworld_env2
from agent.q_learning_agent import ValueIteration
from data_generation.generate_data import GridWorldMDPDataGenerator
from reward_learning.pbirl import PBIRL
from utils.common_helper import (calculate_percentage_optimal_actions,
                                 compute_policy_loss_avar_bound,
                                 calculate_expected_value_difference,)
from utils.env_helper import print_policy

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='Experiment Settings')
parser.add_argument('--num_demonstration', type=int, help='Number of demonstrations', required=True)
parser.add_argument('--beta', type=float, help='beta', required=True)
parser.add_argument('--save_dir', type=str, help='Directory to save results', required=True)

#parser.add_argument('--log_file', type=str, help='Path to the log file', required=False)
args = parser.parse_args()

# Set the save directory in the parent folder's "results" directory
save_dir = os.path.join(parent, "results", args.save_dir)
os.makedirs(save_dir, exist_ok=True)

# Set up the log file path
log_file = os.path.join(save_dir, "default_log.log")

# Set up logging without date and time
logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
logger = logging.getLogger()

# Load config file
with open(parent+"/configs/gridworld_config.yaml", 'r') as file:
    config = yaml.safe_load(file)
logger.info("Config file loaded successfully.")


# Extract config parameters
render_mode = config['env_config']['render_mode']
size = config['env_config']['size']
noise_prob = config['env_config']['noise_prob']
seed = config['env_config']['seed']
gamma = config['env_config']['gamma']
epsilon = float(config['algorithm_config']['epsilon'])

num_steps = config['bayesian_irl_config']['num_steps']
step_stdev = config['bayesian_irl_config']['step_stdev']
beta = beta = float(args.beta)
normalize = config['bayesian_irl_config']['normalize']
adaptive = config['bayesian_irl_config']['adaptive']
burn_frac = config['bayesian_irl_config']['burn_frac']
skip_rate = config['bayesian_irl_config']['skip_rate']

alphas = config['suff_config']['alphas']
delta = config['suff_config']['delta']
optimality_threshold = config['suff_config']['optimality_threshold']
random_normalization = config['suff_config']['random_normalization']
thresholds = config['suff_config']['thresholds']

# Get values from argparse or fallback to YAML config
num_world = config['experiments']['num_world']
num_demonstration = args.num_demonstration if args.num_demonstration else config['experiments']['num_demonstration']

logger.info(f"Running experiment with {num_world} worlds and {num_demonstration} demonstrations per world.")


color_to_feature_map = {
    "red": [1, 0, 0],
    "blue": [0, 1, 0],
    "black": [0, 0, 1]  # 'black' indicates a terminal state
}

custom_grid_features = [
    ["blue", "red", "blue"],
    ["blue", "blue", "black"]
]

# Initialize environments
# Define your feature weights list
#feature_weights_list = [[-0.69171446, -0.20751434,  0.69171446]]
feature_weights_list = np.load("grid_world_weights.npy")

# Initialize environments with feature weights
envs = [gridworld_env2.NoisyLinearRewardFeaturizedGridWorldEnv(gamma=gamma,
    color_to_feature_map=color_to_feature_map,
    grid_features=custom_grid_features,
    custom_feature_weights=list(feat)) for feat in feature_weights_list]

for env in envs:
    logger.info(f"Feature weights for environment: {env.feature_weights}")

# Generate policies for each environment
policies = [ValueIteration(envs[i]).get_optimal_policy() for i in range(num_world)]
print_policy(policies[0], 2, 3)
logger.info(f"Generated optimal policies for all environments.")

# Initialize metrics storage
pairwise_comparisons = [([(0,1), (3,3), (4,3), (5,None)], [(0,1), (3,0), (0,1), (3,0)]), 
                        ([(0,1), (3,3), (4,3), (5,None)], [(0,3), (1,3), (2,1), (5,None)]),
                       ([(0,1), (3,3), (4,3), (5,None)], [(0,3), (1,1), (4,3), (5,None)])]

bounds_all_experiments = []
num_demos_all_experiments = []
true_evds_all_experiments = []
avg_bound_errors_all_experiments = []
policy_optimalities_all_experiments = []
confusion_matrices_all_experiments = []
avar_bound_all_experiments = []
true_avar_bounds_all_experiments = []
#info_gain_all_experiments = []

# Run experiments for each world
for i in range(50):
    env = envs[i]
    logger.info(f"\nRunning experiment {i+1}/{50}...")

    # Shuffle the pairwise comparisons
    pairwise_comparisons_shuffled = pairwise_comparisons.copy()
    random.shuffle(pairwise_comparisons_shuffled)
    logger.info(f"Shuffled pairwise comparisons: {pairwise_comparisons_shuffled}")

    # Initialize metrics for the current experiment
    bounds = {threshold: [] for threshold in thresholds}
    num_demos = {threshold: [] for threshold in thresholds}
    true_evds = {threshold: [] for threshold in thresholds}
    avg_bound_errors = {threshold: [] for threshold in thresholds}
    policy_optimalities = {threshold: [] for threshold in thresholds}
    confusion_matrices = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}

    avar_bounds = {k: {i: [] for i in range(0, num_demonstration)} for k in alphas}
    true_avar_bounds = {i: [] for i in range(0, num_demonstration)}

    #info_gain = {i: [] for i in range(0, num_demonstration)}

    # Run PBIRL for each demonstration
    for demonstration in range(num_demonstration):
        logger.info(f"\nRunning PBIRL with {demonstration + 1} demonstrations for experiment {i+1}")
        print(pairwise_comparisons_shuffled[:demonstration+1])
        birl = PBIRL(env, pairwise_comparisons_shuffled[:demonstration+1], beta)
        birl.run_mcmc(num_steps, step_stdev, adaptive=adaptive)
        logger.info(f"MCMC completed with acceptance ratio: {birl.accept_rate:.4f}")

        burn_indx = int(len(birl.chain) * burn_frac)
        mcmc_samples = birl.chain[burn_indx::]
        logger.info(f"Using {len(mcmc_samples)} samples after burn-in.")

        # Get MAP solution
        map_env = copy.deepcopy(env)
        map_solution = birl.get_map_solution()
        logger.info(f"MAP Solution: {map_solution}")
        logger.info(f"True reward weights: {env.feature_weights}")
        
        map_env.set_feature_weights(map_solution)
        map_policy = ValueIteration(map_env).get_optimal_policy()

        # Visualize the map policy
        logger.info("MAP Policy for current environment:")
        print_policy(map_policy, 2, 3)

        # Calculate a-VaR for different alphas
        for alpha in alphas:
            avar_bound = compute_policy_loss_avar_bound(mcmc_samples, env, map_policy, random_normalization, alpha, delta)
            avar_bounds[alpha][demonstration].append(avar_bound)
            logger.info(f"{alpha}-VaR-max-normalization for {demonstration + 1} demonstrations: {avar_bound:.6f}")

        # Calculate true expected value difference (EVD)
        true_bound = calculate_expected_value_difference(eval_policy=map_policy, env=env, epsilon=epsilon, normalize_with_random_policy=random_normalization)
        true_avar_bounds[demonstration].append(true_bound)
        logger.info(f"True EVD for {demonstration + 1} demonstrations: {true_bound:.6f}")

        # Calculate Information gain
        #infogain = compute_infogain_log(env, pairwise_comparisons_shuffled[demonstration], mcmc_samples, beta)
        #info_gain[demonstration].append(infogain)
        #logger.info(f"Information gain {demonstration + 1} demonstrations: {infogain :.6f}")

        # Check sufficiency with threshold
        for threshold in thresholds:
            avar_bound = avar_bounds[0.95][demonstration][0]
            logger.info(f"Avar bound for threshold {threshold}: {avar_bound:.6f}")
            if avar_bound < threshold:
                logger.info(f"SUFFICIENT ({avar_bound:.6f} < {threshold})")
                map_evd = true_bound
                bounds[threshold].append(avar_bound)
                num_demos[threshold].append(demonstration + 1)
                true_evds[threshold].append(map_evd)
                avg_bound_errors[threshold].append(avar_bound - map_evd)
                policy_optimalities[threshold].append(calculate_percentage_optimal_actions(map_policy, env))
                if true_bound < threshold:
                    confusion_matrices[threshold][0][0] += 1
                else:
                    confusion_matrices[threshold][0][1] += 1
            else:
                logger.info(f"INSUFFICIENT ({avar_bound:.6f} >= {threshold})")
                if true_bound < threshold:
                    confusion_matrices[threshold][1][0] += 1
                else:
                    confusion_matrices[threshold][1][1] += 1

    # Store results for the current experiment
    avar_bound_all_experiments.append(avar_bounds)
    true_avar_bounds_all_experiments.append(true_avar_bounds)
    bounds_all_experiments.append(bounds)
    num_demos_all_experiments.append(num_demos)
    true_evds_all_experiments.append(true_evds)
    avg_bound_errors_all_experiments.append(avg_bound_errors)
    policy_optimalities_all_experiments.append(policy_optimalities)
    confusion_matrices_all_experiments.append(confusion_matrices)
    #info_gain_all_experiments.append(info_gain)

    if (i+1)%2 == 0:
        # Save results to files
        logger.info("\nSaving results to files...")
        np.save(os.path.join(save_dir, 'avar_bound_all_experiments.npy'), avar_bound_all_experiments)
        np.save(os.path.join(save_dir, 'true_avar_bounds_all_experiments.npy'), true_avar_bounds_all_experiments)
        np.save(os.path.join(save_dir, 'bounds_all_experiments.npy'), bounds_all_experiments)
        np.save(os.path.join(save_dir, 'num_demos_all_experiments.npy'), num_demos_all_experiments)
        np.save(os.path.join(save_dir, 'true_evds_all_experiments.npy'), true_evds_all_experiments)
        np.save(os.path.join(save_dir, 'avg_bound_errors_all_experiments.npy'), avg_bound_errors_all_experiments)
        np.save(os.path.join(save_dir, 'policy_optimalities_all_experiments.npy'), policy_optimalities_all_experiments)
        np.save(os.path.join(save_dir, 'confusion_matrices_all_experiments.npy'), confusion_matrices_all_experiments)
        #np.save(os.path.join(save_dir, 'info_gain_all_experiments.npy'), info_gain_all_experiments)
            

        logger.info("Results saved successfully.")