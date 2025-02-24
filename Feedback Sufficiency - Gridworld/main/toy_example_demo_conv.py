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
from reward_learning.birl import BIRL
from utils.common_helper import (calculate_percentage_optimal_actions,
                                 calculate_expected_value_difference,
                                 calculate_policy_accuracy,)
from utils.env_helper import print_policy

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='Experiment Settings')
parser.add_argument('--num_demonstration', type=int, help='Number of demonstrations', required=True)
parser.add_argument('--beta', type=float, help='beta', required=False)
parser.add_argument('--save_dir', type=str, help='Directory to save results', required=True)

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
beta = float(args.beta) if args.beta else config['bayesian_irl_config']['beta']
normalize = config['bayesian_irl_config']['normalize']
adaptive = config['bayesian_irl_config']['adaptive']
burn_frac = config['bayesian_irl_config']['burn_frac']
skip_rate = config['bayesian_irl_config']['skip_rate']

alphas = config['suff_config']['alphas']
delta = config['suff_config']['delta']
optimality_threshold = config['suff_config']['optimality_threshold']
random_normalization = config['suff_config']['random_normalization']

conv_thresholds = config['suff_config']['conv_thresholds']
#num_world = config['experiments']['num_world']
num_demonstration = args.num_demonstration if args.num_demonstration else config['experiments']['num_demonstration']

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
policies = [ValueIteration(envs[i]).get_optimal_policy() for i in range(len(envs))]
logger.info(f"Generated optimal policies for all environments.")

# Initialize metrics storage
demos = [(0, 1), (2, 1), (4, 3)]


########################################################################
# Metrics for convergence - all experiments
########################################################################
num_demos_conv_all = []
pct_states_conv_all = []
policy_optimalities_conv_all = []
policy_accuracies_conv_all = []
accuracies_conv_all = []
cm100_conv_all = []
cm95_conv_all = []
cm90_conv_all = []
cm5_conv_all = []
cm4_conv_all = []
cm3_conv_all = []
cm2_conv_all = []
cm1_conv_all = []


# Initialize MCMC storage
mcmc_samples_all_experiments = {}  # Track MCMC samples across experiments
same_demonstration = True

# Run experiments for each world
for i in range(50):
    env = envs[i]
    logger.info(f"\nRunning experiment {i+1}/{50}...")

    if same_demonstration:
            # Randomly select one pairwise comparison and replicate it
        selected_demo = random.choice(demos)
        demos_shuffled = [selected_demo] * num_demonstration  # Replicate the chosen demo
        logger.info(f"Same Demos: {demos_shuffled}")

    else:
        # Shuffle the pairwise comparisons
        demos_shuffled = demos.copy()
        random.shuffle(demos_shuffled)
        logger.info(f"Shuffled Demos: {demos_shuffled}")

    ########################################################################
    # Metrics for convergence
    ########################################################################
    num_demos_conv = {threshold: [] for threshold in conv_thresholds}
    pct_states_conv = {threshold: [] for threshold in conv_thresholds}
    policy_optimalities_conv = {threshold: [] for threshold in conv_thresholds}
    policy_accuracies_conv = {threshold: [] for threshold in conv_thresholds}
    accuracies_conv = {threshold: [] for threshold in conv_thresholds}
    # Predicted by true
    cm100_conv = {threshold: [[0, 0], [0, 0]] for threshold in conv_thresholds}  # actual positive is if optimality = 100%/99%
    cm95_conv = {threshold: [[0, 0], [0, 0]] for threshold in conv_thresholds}  # actual positive is if optimality = 95%/96%
    cm90_conv = {threshold: [[0, 0], [0, 0]] for threshold in conv_thresholds}  # actual positive is if optimality = 90%/92%
    cm5_conv = {threshold: [[0, 0], [0, 0]] for threshold in conv_thresholds}  # actual positive is with nEVD threshold = 0.5
    cm4_conv = {threshold: [[0, 0], [0, 0]] for threshold in conv_thresholds}  # actual positive is with nEVD threshold = 0.4
    cm3_conv = {threshold: [[0, 0], [0, 0]] for threshold in conv_thresholds}  # actual positive is with nEVD threshold = 0.3
    cm2_conv = {threshold: [[0, 0], [0, 0]] for threshold in conv_thresholds}  # actual positive is with nEVD threshold = 0.2
    cm1_conv = {threshold: [[0, 0], [0, 0]] for threshold in conv_thresholds}  # actual positive is with nEVD threshold = 0.1
    
    curr_map_pi = [-1 for _ in range(env.rows * env.columns)]
    patience = 0
    
    ########################################################################
    # Metrics for held-out
    ########################################################################

    # Run PBIRL for each demonstration
    for demonstration in range(num_demonstration):

        logger.info(f"\nRunning PBIRL with {demonstration + 1} demonstrations for experiment {i+1}")
        print(demos_shuffled[:demonstration+1])
        birl = BIRL(env, demos_shuffled[:demonstration+1], beta)
        birl.run_mcmc(num_steps, step_stdev, adaptive=adaptive)
        logger.info(f"MCMC completed with acceptance ratio: {birl.accept_rate:.4f}")

        burn_indx = int(len(birl.chain) * burn_frac)
        mcmc_samples = birl.chain[burn_indx::]
        logger.info(f"Using {len(mcmc_samples)} samples after burn-in.")

        # Save MCMC samples history
        logger.info(f"Stored {len(mcmc_samples)} MCMC samples for demonstration {demonstration+1}")

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
        ########################################################################
        # Convergence
        ########################################################################
    
        policy_match = calculate_policy_accuracy(curr_map_pi, map_policy)
    
        if policy_match == 1.0:
            patience += 1
            # evaluate thresholds
            for t in range(len(conv_thresholds)):
                threshold = conv_thresholds[t]
                actual_nevd = calculate_expected_value_difference(eval_policy=map_policy, env=env, epsilon=epsilon, normalize_with_random_policy=random_normalization)
                optimality = calculate_percentage_optimal_actions(map_policy, env)
                if patience == threshold:
                    num_demos_conv[threshold].append(demonstration + 1)
                    pct_states_conv[threshold].append((demonstration + 1) / (env.rows * env.columns))
                    policy_optimalities_conv[threshold].append(optimality)
                    policy_accuracies_conv[threshold].append(calculate_policy_accuracy(policies[i], map_policy))
                    accuracies_conv[threshold].append(optimality >= optimality_threshold)
                    curr_map_pi = map_policy
    
                    # Evaluate actual positive by optimality
                    if optimality >= 1.0:
                        cm100_conv[threshold][0][0] += 1
                    else:
                        cm100_conv[threshold][0][1] += 1
                    if optimality >= 0.95:
                        cm95_conv[threshold][0][0] += 1
                    else:
                        cm95_conv[threshold][0][1] += 1
                    if optimality >= 0.90:
                        cm90_conv[threshold][0][0] += 1
                    else:
                        cm90_conv[threshold][0][1] += 1
                    # Evaluate actual positive by nEVD
                    if actual_nevd < 0.5:
                        cm5_conv[threshold][0][0] += 1
                    else:
                        cm5_conv[threshold][0][1] += 1
                    if actual_nevd < 0.4:
                        cm4_conv[threshold][0][0] += 1
                    else:
                        cm4_conv[threshold][0][1] += 1
                    if actual_nevd < 0.3:
                        cm3_conv[threshold][0][0] += 1
                    else:
                        cm3_conv[threshold][0][1] += 1
                    if actual_nevd < 0.2:
                        cm2_conv[threshold][0][0] += 1
                    else:
                        cm2_conv[threshold][0][1] += 1
                    if actual_nevd < 0.1:
                        cm1_conv[threshold][0][0] += 1
                    else:
                        cm1_conv[threshold][0][1] += 1
                else:
                    # Evaluate actual positive by optimality
                    if optimality >= 1.0:
                        cm100_conv[threshold][1][0] += 1
                    else:
                        cm100_conv[threshold][1][1] += 1
                    if optimality >= 0.95:
                        cm95_conv[threshold][1][0] += 1
                    else:
                        cm95_conv[threshold][1][1] += 1
                    if optimality >= 0.90:
                        cm90_conv[threshold][1][0] += 1
                    else:
                        cm90_conv[threshold][1][1] += 1
                    # Evaluate actual positive by nEVD
                    if actual_nevd < 0.5:
                        cm5_conv[threshold][1][0] += 1
                    else:
                        cm5_conv[threshold][1][1] += 1
                    if actual_nevd < 0.4:
                        cm4_conv[threshold][1][0] += 1
                    else:
                        cm4_conv[threshold][1][1] += 1
                    if actual_nevd < 0.3:
                        cm3_conv[threshold][1][0] += 1
                    else:
                        cm3_conv[threshold][1][1] += 1
                    if actual_nevd < 0.2:
                        cm2_conv[threshold][1][0] += 1
                    else:
                        cm2_conv[threshold][1][1] += 1
                    if actual_nevd < 0.1:
                        cm1_conv[threshold][1][0] += 1
                    else:
                        cm1_conv[threshold][1][1] += 1
        else:
            patience = 0
            curr_map_pi = map_policy

    ########################################################################
    # Store results for Convergence
    ########################################################################
    num_demos_conv_all.append(num_demos_conv)
    pct_states_conv_all.append(pct_states_conv)
    policy_optimalities_conv_all.append(policy_optimalities_conv)
    policy_accuracies_conv_all.append(policy_accuracies_conv)
    accuracies_conv_all.append(accuracies_conv)
    cm100_conv_all.append(cm100_conv)
    cm95_conv_all.append(cm95_conv)
    cm90_conv_all.append(cm90_conv)
    cm5_conv_all.append(cm5_conv)
    cm4_conv_all.append(cm4_conv)
    cm3_conv_all.append(cm3_conv)
    cm2_conv_all.append(cm2_conv)
    cm1_conv_all.append(cm1_conv)



    if (i+1)%2 == 0:
        # Save results to files
        ########################################################################
        # Save Convergence results in file
        ########################################################################
        logger.info("\nSaving results to files...")
        np.save(os.path.join(save_dir, 'num_demos_conv_all.npy'), num_demos_conv_all)
        np.save(os.path.join(save_dir, 'pct_states_conv_all.npy'), pct_states_conv_all)
        np.save(os.path.join(save_dir, 'policy_optimalities_conv_all.npy'), policy_optimalities_conv_all)
        np.save(os.path.join(save_dir, 'policy_accuracies_conv_all.npy'), policy_accuracies_conv_all)
        np.save(os.path.join(save_dir, 'accuracies_conv_all.npy'), accuracies_conv_all)
        np.save(os.path.join(save_dir, 'cm100_conv_all.npy'), cm100_conv_all)
        np.save(os.path.join(save_dir, 'cm95_conv_all.npy'), cm95_conv_all)
        np.save(os.path.join(save_dir, 'cm90_conv_all.npy'), cm90_conv_all)
        np.save(os.path.join(save_dir, 'cm5_conv_all.npy'), cm5_conv_all)
        np.save(os.path.join(save_dir, 'cm4_conv_all.npy'), cm4_conv_all)
        np.save(os.path.join(save_dir, 'cm3_conv_all.npy'), cm3_conv_all)
        np.save(os.path.join(save_dir, 'cm2_conv_all.npy'), cm2_conv_all)
        np.save(os.path.join(save_dir, 'cm1_conv_all.npy'), cm1_conv_all)
        logger.info("Results saved successfully.")
