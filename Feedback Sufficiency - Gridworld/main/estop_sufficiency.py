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

from env import gridworld_env
from agent.q_learning_agent import ValueIteration
from data_generation.generate_data import GridWorldMDPDataGenerator, generate_random_trajectory, simulate_human_estop, simulate_human_estop_v2
from reward_learning.ebirl_v2 import EBIRL
from utils.common_helper import (calculate_percentage_optimal_actions,
                                 compute_policy_loss_avar_bounds,
                                 calculate_expected_value_difference)
from utils.env_helper import print_policy_2

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='Experiment Settings')
parser.add_argument('--num_demonstration', type=int, help='Number of demonstrations', required=False)
parser.add_argument('--save_dir', type=str, help='Directory to save results', required=False)
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
beta = config['bayesian_irl_config']['beta']
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
#num_world = config['experiments']['num_world']
num_demonstration = args.num_demonstration if args.num_demonstration else config['experiments']['num_demonstration']

#logger.info(f"Running experiment with {num_world} worlds and {num_demonstration} demonstrations per world.")

# Initialize environments
# Define your feature weights list
feature_weights_list = [
[0.0931973601692278, 0.31955139750932166, 0.8025237136446507, 1.2150678890096278],
[-1.043896398189713, -0.4942330908379698, -0.226273043977835, 2.1048164480968876],
[-0.17418442541958648, -0.003489576892883923, 0.32593002952435973, 0.35269980857079075],
[-0.9280931904508574, -0.2881080257028875, 0.08962292282477151, 1.052187184051626],
[-1.7231474630133796, -0.6988329431080711, 0.5472225722003625, 1.5488623949444609],
[-1.0209494398790047, -0.6854443566443772, -0.6476753625995035, 1.0385550023579349],
[-1.7845182158204707, -1.6787110627632869, 0.8288941048913488, 1.2115616873217196],
[-2.174978537296403, -1.0972226428986527, -0.31426035106803707, 0.31718860993597364],
[-1.3913605343466946, -0.5824467367040048, 0.8644205766780406, 1.1809629221011249],
[-1.975898745939323, -1.355090466173791, 1.0061202083382523, 1.0130583574165686],
[-1.156602653220867, -0.31576595791368006, 0.720789911816225, 1.349683775061085],
[-0.12465340180602934, -0.11618654388446568, 0.2564478770411658, 0.44843665020275864],
[-2.675729984600984, -1.271465736774545, 2.1974242730661526, 2.695323060703398],
[-1.5168053078904342, -0.3907796306017249, 0.06451276795927917, 0.20296366497020824],
[-0.5883530759377323, 0.1371846522458069, 1.0320495436626964, 1.7824479661009536],
[-1.5972542927197892, -1.4669496080110351, -0.27573782848273937, 0.271535946685796],
[-1.5034759991154618, 0.011924000946604054, 0.0943121849295515, 0.30178168103031955],
[-1.0116227897104013, 0.07027963552571587, 1.1928121939469933, 2.447968680112173],
[-1.261642036698053, 0.373183013830343, 0.38646698620288805, 0.5377604226071607],
[-1.5018118880850853, 0.32039581205828177, 1.0648859103194126, 1.9467375399649056],
]
num_world = len(feature_weights_list)

# Initialize environments with feature weights
envs = [gridworld_env.NoisyLinearRewardFeaturizedGridWorldEnv(gamma=gamma, size=size, noise_prob=noise_prob) 
        for _ in range(len(feature_weights_list))]

# Loop through each environment and set feature weights
for env, weights in zip(envs, feature_weights_list):
    env.set_feature_weights(weights)

for env in envs:
    logger.info(f"Feature weights for environment: {env.feature_weights}")

logger.info(f"Initialized {num_world} GridWorld environments.")

# Generate policies for each environment
policies = [ValueIteration(envs[i]).get_optimal_policy() for i in range(num_world)]
logger.info(f"Generated optimal policies for all environments.")
print_policy_2(policies[0], 3)

# Initialize metrics storage
demo_order = list(range(size * size))
random.shuffle(demo_order)
logger.info(f"Shuffled demonstration order: {demo_order}")

bounds_all_experiments = []
num_demos_all_experiments = []
true_evds_all_experiments = []
avg_bound_errors_all_experiments = []
policy_optimalities_all_experiments = []
confusion_matrices_all_experiments = []
avar_bound_all_experiments = []
true_avar_bounds_all_experiments = []

# Run experiments for each world
for i in range(num_world):
    env = envs[i]
    logger.info(f"\nRunning experiment for environment {i+1}/{num_world}...")

    random_trajs = [generate_random_trajectory(env, max_horizon=20) for j in range(15)]
    #random_trajs = [generate_random_trajectory_diff_start(envs[i], max_horizon=5) for j in range(6)]

    #estop_demo = [simulate_human_estop(env, j, beta=beta, gamma=gamma, fixed_length=None) for j in random_trajs]

    estop_demo = [simulate_human_estop_v2(env, j, beta=beta, gamma=gamma) for j in random_trajs]

    #estop_demo = GridWorldMDPDataGenerator(env=env, seed=seed).generate_estop(beta=beta, num_trajs=num_demonstration)
    logger.info(f"Generated Estops for {num_demonstration} demonstrations.")

    # Initialize metrics for the current experiment
    bounds = {threshold: [] for threshold in thresholds}
    num_demos = {threshold: [] for threshold in thresholds}
    true_evds = {threshold: [] for threshold in thresholds}
    avg_bound_errors = {threshold: [] for threshold in thresholds}
    policy_optimalities = {threshold: [] for threshold in thresholds}
    confusion_matrices = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}

    avar_bounds = {k: {i: [] for i in range(0, num_demonstration)} for k in alphas}
    true_avar_bounds = {i: [] for i in range(0, num_demonstration)}

    # Run PBIRL for each demonstration
    for demonstration in range(len(estop_demo)):
        logger.info(f"\nRunning PBIRL with {demonstration + 1} demonstrations for environment {i+1}")

        birl = EBIRL(env, estop_demo[:demonstration+1], beta)
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
        #print_policy(map_policy, size)
        print_policy_2(map_policy, 3)

        approx_avar_bounds = compute_policy_loss_avar_bounds(mcmc_samples, env, map_policy, random_normalization, alphas, delta)

        # Calculate a-VaR for different alphas
        for alpha in alphas:
            #avar_bound = compute_policy_loss_avar_bound(mcmc_samples, env, map_policy, random_normalization, alpha, delta)
            avar_bounds[alpha][demonstration].append(approx_avar_bounds[alpha])
            logger.info(f"{alpha}-VaR-max-normalization for {demonstration + 1} demonstrations: {approx_avar_bounds[alpha]:.6f}")

        # Calculate true expected value difference (EVD)
        true_bound = calculate_expected_value_difference(eval_policy=map_policy, env=env, epsilon=epsilon, normalize_with_random_policy=random_normalization)
        true_avar_bounds[demonstration].append(true_bound)
        logger.info(f"True EVD for {demonstration + 1} demonstrations: {true_bound:.6f}")

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

        logger.info("Results saved successfully.")