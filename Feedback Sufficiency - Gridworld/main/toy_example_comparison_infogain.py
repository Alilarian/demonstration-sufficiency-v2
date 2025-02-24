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
from reward_learning.pbirl import PBIRL
from utils.common_helper import (calculate_percentage_optimal_actions,
                                 calculate_expected_value_difference,
                                 calculate_policy_accuracy,
                                compute_infogain_7,
                                log_prob_comparison)
from utils.env_helper import print_policy

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='Experiment Settings')
parser.add_argument('--num_demonstration', type=int, help='Number of demonstrations', required=True)
parser.add_argument('--beta', type=float, help='beta', required=False)
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
beta = float(args.beta) if args.beta else config['bayesian_irl_config']['beta']
normalize = config['bayesian_irl_config']['normalize']
adaptive = config['bayesian_irl_config']['adaptive']
burn_frac = config['bayesian_irl_config']['burn_frac']
skip_rate = config['bayesian_irl_config']['skip_rate']

alphas = config['suff_config']['alphas']
delta = config['suff_config']['delta']
optimality_threshold = config['suff_config']['optimality_threshold']
random_normalization = config['suff_config']['random_normalization']
infogain_thresholds = [0.001, 0.0014, 0.00196, 0.002744, 0.0038415999999999997, 0.005378239999999999, 0.007529535999999999, 0.010541350399999998, 0.014757890559999997, 0.020661046783999996, 0.028925465497599993, 0.04049565169663999, 0.05669391237529598, 0.07937147732541437, 0.11112006825558012, 0.15556809555781215, 0.217795333780937, 0.30491346729331176, 0.4268788542106364, 0.597630395894891, 0.8366825542528473, 1.1713555759539862, 1.6398978063355807, 2.2958569288698127, 3.214199700417738, 4.499879580584833, 6.299831412818765, 8.81976397794627]

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

# Loop through each environment and set feature weights
#for env, weights in zip(envs, feature_weights_list):
#    env.set_feature_weights(weights)

for env in envs:
    logger.info(f"Feature weights for environment: {env.feature_weights}")

# Generate policies for each environment
policies = [ValueIteration(envs[i]).get_optimal_policy() for i in range(len(envs))]
logger.info(f"Generated optimal policies for all environments.")

# Initialize metrics storage
pairwise_comparisons = [([(0,1), (3,3), (4,3), (5,None)], [(0,1), (3,0), (0,1), (3,0)]), 
                        ([(0,1), (3,3), (4,3), (5,None)], [(0,3), (1,3), (2,1), (5,None)]),
                       ([(0,1), (3,3), (4,3), (5,None)], [(0,3), (1,1), (4,3), (5,None)])]

########################################################################
# Metrics for infogainergence - all experiments
########################################################################
num_demos_infogain_all = []
pct_states_infogain_all = []
policy_optimalities_infogain_all = []
policy_accuracies_infogain_all = []
accuracies_infogain_all = []
info_gain_all_experiments = []
cm100_infogain_all = []
cm95_infogain_all = []
cm90_infogain_all = []
cm5_infogain_all = []
cm4_infogain_all = []
cm3_infogain_all = []
cm2_infogain_all = []
cm1_infogain_all = []




# Initialize MCMC storage
mcmc_samples_all_experiments = {}  # Track MCMC samples across experiments
same_demonstration = True

# Run experiments for each world
for i in range(50):
    env = envs[i]
    logger.info(f"\nRunning experiment {i+1}/{50}...")

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

    #previous_mcmc_samples = None  # Initial prior: None (Uniform)

    ########################################################################
    # Metrics to evaluate nEVD
    ########################################################################


    ########################################################################
    # Metrics for infogain
    ########################################################################
    num_demos_infogain = {threshold: [] for threshold in infogain_thresholds}
    pct_states_infogain = {threshold: [] for threshold in infogain_thresholds}
    policy_optimalities_infogain = {threshold: [] for threshold in infogain_thresholds}
    policy_accuracies_infogain = {threshold: [] for threshold in infogain_thresholds}
    accuracies_infogain = {threshold: [] for threshold in infogain_thresholds}
    # Predicted by true
    cm100_infogain = {threshold: [[0, 0], [0, 0]] for threshold in infogain_thresholds}  # actual positive is if optimality = 100%/99%
    cm95_infogain = {threshold: [[0, 0], [0, 0]] for threshold in infogain_thresholds}  # actual positive is if optimality = 95%/96%
    cm90_infogain = {threshold: [[0, 0], [0, 0]] for threshold in infogain_thresholds}  # actual positive is if optimality = 90%/92%
    cm5_infogain = {threshold: [[0, 0], [0, 0]] for threshold in infogain_thresholds}  # actual positive is with nEVD threshold = 0.5
    cm4_infogain = {threshold: [[0, 0], [0, 0]] for threshold in infogain_thresholds}  # actual positive is with nEVD threshold = 0.4
    cm3_infogain = {threshold: [[0, 0], [0, 0]] for threshold in infogain_thresholds}  # actual positive is with nEVD threshold = 0.3
    cm2_infogain = {threshold: [[0, 0], [0, 0]] for threshold in infogain_thresholds}  # actual positive is with nEVD threshold = 0.2
    cm1_infogain = {threshold: [[0, 0], [0, 0]] for threshold in infogain_thresholds}  # actual positive is with nEVD threshold = 0.1
    

    info_gain_per_demo = {i: [] for i in range(0, num_demonstration)}
    ########################################################################
    # Metrics for held-out
    ########################################################################

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
        mcmc_samples_history[demonstration + 1] = mcmc_samples
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

        if demonstration == 0:
            mcmc_sample_prior = []
            outer_mcmc_samples_prior = []

        else:
            mcmc_sample_prior = [sample for i in range(1, demonstration+1) for sample in mcmc_samples_history[i]]
            outer_mcmc_samples_prior = mcmc_samples_history[demonstration]

        
        posterior_mcmc_samples = [sample for i in range(1, demonstration+2) for sample in mcmc_samples_history[i]]
        
        print("Len prior samples: ", len(mcmc_sample_prior))
        print("Len posterior samples: ", len(posterior_mcmc_samples))
        
        infogain = compute_infogain_7(env,
                       demos=demos_shuffled[:demonstration+1], 
                       inner_mcmc_samples_prior=mcmc_sample_prior, 
                       inner_mcmc_samples_post=posterior_mcmc_samples,
                       outer_mcmc_samples_prior=outer_mcmc_samples_prior,
                       outer_mcmc_samples_post=mcmc_samples_history[demonstration+1],
                        beta=beta, 
                        log_prob_func=log_prob_comparison)

        info_gain_per_demo[demonstration].append(infogain)
        logger.info(f"Information gain {demonstration + 1} demonstrations: {infogain :.6f}")
        
        # evaluate thresholds
        for t in range(len(infogain_thresholds)):
            threshold = infogain_thresholds[t]
            actual_nevd = calculate_expected_value_difference(eval_policy=map_policy, env=env, epsilon=epsilon, normalize_with_random_policy=random_normalization)
            optimality = calculate_percentage_optimal_actions(map_policy, env)
            if infogain <= threshold:
                num_demos_infogain[threshold].append(demonstration + 1)
                pct_states_infogain[threshold].append((demonstration + 1) / (env.rows * env.columns))
                policy_optimalities_infogain[threshold].append(optimality)
                policy_accuracies_infogain[threshold].append(calculate_policy_accuracy(policies[i], map_policy))
                accuracies_infogain[threshold].append(optimality >= optimality_threshold)
                curr_map_pi = map_policy

                # Evaluate actual positive by optimality
                if optimality >= 1.0:
                    cm100_infogain[threshold][0][0] += 1
                else:
                    cm100_infogain[threshold][0][1] += 1
                if optimality >= 0.95:
                    cm95_infogain[threshold][0][0] += 1
                else:
                    cm95_infogain[threshold][0][1] += 1
                if optimality >= 0.90:
                    cm90_infogain[threshold][0][0] += 1
                else:
                    cm90_infogain[threshold][0][1] += 1
                # Evaluate actual positive by nEVD
                if actual_nevd < 0.5:
                    cm5_infogain[threshold][0][0] += 1
                else:
                    cm5_infogain[threshold][0][1] += 1
                if actual_nevd < 0.4:
                    cm4_infogain[threshold][0][0] += 1
                else:
                    cm4_infogain[threshold][0][1] += 1
                if actual_nevd < 0.3:
                    cm3_infogain[threshold][0][0] += 1
                else:
                    cm3_infogain[threshold][0][1] += 1
                if actual_nevd < 0.2:
                    cm2_infogain[threshold][0][0] += 1
                else:
                    cm2_infogain[threshold][0][1] += 1
                if actual_nevd < 0.1:
                    cm1_infogain[threshold][0][0] += 1
                else:
                    cm1_infogain[threshold][0][1] += 1
            else:
                # Evaluate actual positive by optimality
                if optimality >= 1.0:
                    cm100_infogain[threshold][1][0] += 1
                else:
                    cm100_infogain[threshold][1][1] += 1
                if optimality >= 0.95:
                    cm95_infogain[threshold][1][0] += 1
                else:
                    cm95_infogain[threshold][1][1] += 1
                if optimality >= 0.90:
                    cm90_infogain[threshold][1][0] += 1
                else:
                    cm90_infogain[threshold][1][1] += 1
                # Evaluate actual positive by nEVD
                if actual_nevd < 0.5:
                    cm5_infogain[threshold][1][0] += 1
                else:
                    cm5_infogain[threshold][1][1] += 1
                if actual_nevd < 0.4:
                    cm4_infogain[threshold][1][0] += 1
                else:
                    cm4_infogain[threshold][1][1] += 1
                if actual_nevd < 0.3:
                    cm3_infogain[threshold][1][0] += 1
                else:
                    cm3_infogain[threshold][1][1] += 1
                if actual_nevd < 0.2:
                    cm2_infogain[threshold][1][0] += 1
                else:
                    cm2_infogain[threshold][1][1] += 1
                if actual_nevd < 0.1:
                    cm1_infogain[threshold][1][0] += 1
                else:
                    cm1_infogain[threshold][1][1] += 1


    ########################################################################
    # Store MCMC samples
    ########################################################################
    mcmc_samples_all_experiments[i + 1] = mcmc_samples_history
    

    
    ########################################################################
    # Store results for infogain
    ########################################################################
    num_demos_infogain_all.append(num_demos_infogain)
    pct_states_infogain_all.append(pct_states_infogain)
    policy_optimalities_infogain_all.append(policy_optimalities_infogain)
    policy_accuracies_infogain_all.append(policy_accuracies_infogain)
    accuracies_infogain_all.append(accuracies_infogain)
    cm100_infogain_all.append(cm100_infogain)
    cm95_infogain_all.append(cm95_infogain)
    cm90_infogain_all.append(cm90_infogain)
    cm5_infogain_all.append(cm5_infogain)
    cm4_infogain_all.append(cm4_infogain)
    cm3_infogain_all.append(cm3_infogain)
    cm2_infogain_all.append(cm2_infogain)
    cm1_infogain_all.append(cm1_infogain)    
    info_gain_all_experiments.append(info_gain_per_demo)

    if (i+1)%2 == 0:
        # Save results to files
        logger.info("\nSaving results to files...")
        
        np.save(os.path.join(save_dir, 'num_demos_infogain_all.npy'), num_demos_infogain_all)
        np.save(os.path.join(save_dir, 'pct_states_infogain_all.npy'), pct_states_infogain_all)
        np.save(os.path.join(save_dir, 'policy_optimalities_infogain_all.npy'), policy_optimalities_infogain_all)
        np.save(os.path.join(save_dir, 'policy_accuracies_infogain_all.npy'), policy_accuracies_infogain_all)
        np.save(os.path.join(save_dir, 'accuracies_infogain_all.npy'), accuracies_infogain_all)
        np.save(os.path.join(save_dir, 'cm100_infogain_all.npy'), cm100_infogain_all)
        np.save(os.path.join(save_dir, 'cm95_infogain_all.npy'), cm95_infogain_all)
        np.save(os.path.join(save_dir, 'cm90_infogain_all.npy'), cm90_infogain_all)
        np.save(os.path.join(save_dir, 'cm5_infogain_all.npy'), cm5_infogain_all)
        np.save(os.path.join(save_dir, 'cm4_infogain_all.npy'), cm4_infogain_all)
        np.save(os.path.join(save_dir, 'cm3_infogain_all.npy'), cm3_infogain_all)
        np.save(os.path.join(save_dir, 'cm2_infogain_all.npy'), cm2_infogain_all)
        np.save(os.path.join(save_dir, 'cm1_infogain_all.npy'), cm1_infogain_all)
        np.save(os.path.join(save_dir, 'info_gain_all_experiments.npy'), info_gain_all_experiments)
                
        logger.info("Results saved successfully.")