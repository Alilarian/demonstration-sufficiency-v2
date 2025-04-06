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
from reward_learning.pbirl import PBIRL
from data_generation.generate_data import generate_pairwise_comparisons
from utils.common_helper import (calculate_percentage_optimal_actions,
                                 calculate_expected_value_difference,
                                 calculate_policy_accuracy,
)

from utils.env_helper import print_policy_2

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='Experiment Settings')
parser.add_argument('--num_demonstration', type=int, help='Number of demonstrations', required=True)
parser.add_argument('--save_dir', type=str, help='Directory to save results', required=True)
parser.add_argument('--seed', type=int, required=True)
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
#thresholds = config['suff_config']['thresholds']
conv_thresholds = config['suff_config']['conv_thresholds']

# Get values from argparse or fallback to YAML config
num_demonstration = args.num_demonstration

# Fixing Seeds
random.seed(args.seed)  # Fix Python's built-in random module
np.random.seed(args.seed)  # Fix NumPy
os.environ['PYTHONHASHSEED'] = str(args.seed)  # Ensure deterministic hashing

# Initialize environments
# Define your feature weights list
feature_weights_list = [
[-1.043896398189713, -0.4942330908379698, -0.226273043977835, 2.1048164480968876],
[-0.17418442541958648, -0.003489576892883923, 0.32593002952435973, 0.65269980857079075],
[-0.9280931904508574, -0.2881080257028875, 0.08962292282477151, 1.052187184051626],
[-1.7231474630133796, -0.6988329431080711, 0.5472225722003625, 1.5488623949444609],
[-1.0209494398790047, -0.6854443566443772, -0.6476753625995035, 1.0385550023579349],
[-1.7845182158204707, -1.6787110627632869, 0.8288941048913488, 1.2115616873217196],
[-2.174978537296403, -1.0972226428986527, -0.31426035106803707, 0.51718860993597364],
[-1.3913605343466946, -0.5824467367040048, 0.8644205766780406, 1.1809629221011249],
[-1.975898745939323, -1.355090466173791, 1.0061202083382523, 1.0130583574165686],
[-1.156602653220867, -0.31576595791368006, 0.720789911816225, 1.349683775061085],
[-0.12465340180602934, -0.11618654388446568, 0.2564478770411658, 1.74843665020275864],
[-2.675729984600984, -1.271465736774545, 2.1974242730661526, 2.695323060703398],
[-1.5168053078904342, -0.3907796306017249, 0.06451276795927917, 0.90296366497020824],
[-0.5883530759377323, 0.1371846522458069, 1.0320495436626964, 1.7824479661009536],
[-1.5972542927197892, -1.4669496080110351, -0.27573782848273937, 0.871535946685796],
[-1.5034759991154618, 0.011924000946604054, 0.0943121849295515, 0.60178168103031955],
[-1.0116227897104013, 0.07027963552571587, 1.1928121939469933, 2.447968680112173],
[-1.261642036698053, 0.373183013830343, 0.38646698620288805, 1.1377604226071607],
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

print_policy_2(policies[0], size=size)


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

same_demonstration = False


# Run experiments for each world
for i in range(num_world):
    env = envs[i]
    logger.info(f"\nRunning experiment {i+1}/{num_world}...")

        
    pairwise_comparisons = generate_pairwise_comparisons(env, num_trajs=10, max_horizon=size*size, num_comparisons=num_demonstration)
    
    
    if same_demonstration:
            # Randomly select one pairwise comparison and replicate it
        selected_demo = random.choice(pairwise_comparisons)
        demos_shuffled = [selected_demo] * num_demonstration  # Replicate the chosen demo
        logger.info(f"Same Demos: {demos_shuffled}")

    else:
    
        # Shuffle the pairwise comparisons
        demos_shuffled = pairwise_comparisons.copy()
        random.shuffle(demos_shuffled)
        logger.info(f"Shuffled Demos: {demos_shuffled}")

    # Storage for each experiment
    mcmc_samples_history = {}  # Track MCMC samples for this experiment

    ########################################################################
    # Metrics for entropyConf
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
    
    curr_map_pi = [-1 for _ in range(size * size)]
    patience = 0
    


    # Run PBIRL for each demonstration
    for demonstration in range(num_demonstration):

        logger.info(f"\nRunning PBIRL with {demonstration + 1} demonstrations for experiment {i+1}")
        print(demos_shuffled[:demonstration+1])
        birl = PBIRL(env, demos_shuffled[:demonstration+1], beta)
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
        print_policy_2(map_policy, size)
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
                    pct_states_conv[threshold].append((demonstration + 1) / (size * size))
                    policy_optimalities_conv[threshold].append(optimality)
                    print(i)
                    print(policies[i])
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
