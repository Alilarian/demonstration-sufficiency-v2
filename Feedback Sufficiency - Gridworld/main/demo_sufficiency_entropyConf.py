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
from data_generation.generate_data import GridWorldMDPDataGenerator
from reward_learning.birl import BIRL
from utils.common_helper import (calculate_percentage_optimal_actions,
                                 calculate_expected_value_difference,
                                 calculate_policy_accuracy,
                                compute_entropy,
                                log_prob_demo)
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
entropyConf_thresholds  = [0.001, 0.0014, 0.00196, 0.002744, 0.0038415999999999997, 0.005378239999999999, 0.007529535999999999, 0.010541350399999998, 0.014757890559999997, 0.020661046783999996, 0.028925465497599993, 0.04049565169663999, 0.05669391237529598, 0.07937147732541437, 0.11112006825558012, 0.15556809555781215, 0.217795333780937, 0.30491346729331176, 0.4268788542106364, 0.597630395894891, 0.8366825542528473, 1.1713555759539862, 1.6398978063355807, 2.2958569288698127, 3.214199700417738, 4.499879580584833, 6.299831412818765, 8.81976397794627]


# Get values from argparse or fallback to YAML config
#num_world = config['experiments']['num_world']
num_demonstration = args.num_demonstration if args.num_demonstration else config['experiments']['num_demonstration']

#logger.info(f"Running experiment with {num_world} worlds and {num_demonstration} demonstrations per world.")

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

# Initialize metrics storage
demo_order = list(range(size * size))
random.shuffle(demo_order)
logger.info(f"Shuffled demonstration order: {demo_order}")

########################################################################
# Metrics for entropyConfergence - all experiments
########################################################################
num_demos_entropyConf_all = []
pct_states_entropyConf_all = []
policy_optimalities_entropyConf_all = []
policy_accuracies_entropyConf_all = []
accuracies_entropyConf_all = []
entropyConv_all_experiments = []
cm100_entropyConf_all = []
cm95_entropyConf_all = []
cm90_entropyConf_all = []
cm5_entropyConf_all = []
cm4_entropyConf_all = []
cm3_entropyConf_all = []
cm2_entropyConf_all = []
cm1_entropyConf_all = []

# Initialize MCMC storage
mcmc_samples_all_experiments = {}  # Track MCMC samples across experiments
same_demonstration = True

# Begin world iteration
for i in range(num_world):
    env = envs[i]
    logger.info(f"\nRunning experiment {i+1}/{num_world}...")

    # Randomly pick some states with their optimal demonstrations => I can pick them from policies
    demos = random.sample(policies[i][:-1], k=num_demonstration)

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

 # Storage for each experiment
    mcmc_samples_history = {}  # Track MCMC samples for this experiment

    ########################################################################
    # Metrics for entropyConf
    ########################################################################
    num_demos_entropyConf = {threshold: [] for threshold in entropyConf_thresholds}
    pct_states_entropyConf = {threshold: [] for threshold in entropyConf_thresholds}
    policy_optimalities_entropyConf = {threshold: [] for threshold in entropyConf_thresholds}
    policy_accuracies_entropyConf = {threshold: [] for threshold in entropyConf_thresholds}
    accuracies_entropyConf = {threshold: [] for threshold in entropyConf_thresholds}
    # Predicted by true
    cm100_entropyConf = {threshold: [[0, 0], [0, 0]] for threshold in entropyConf_thresholds}  # actual positive is if optimality = 100%/99%
    cm95_entropyConf = {threshold: [[0, 0], [0, 0]] for threshold in entropyConf_thresholds}  # actual positive is if optimality = 95%/96%
    cm90_entropyConf = {threshold: [[0, 0], [0, 0]] for threshold in entropyConf_thresholds}  # actual positive is if optimality = 90%/92%
    cm5_entropyConf = {threshold: [[0, 0], [0, 0]] for threshold in entropyConf_thresholds}  # actual positive is with nEVD threshold = 0.5
    cm4_entropyConf = {threshold: [[0, 0], [0, 0]] for threshold in entropyConf_thresholds}  # actual positive is with nEVD threshold = 0.4
    cm3_entropyConf = {threshold: [[0, 0], [0, 0]] for threshold in entropyConf_thresholds}  # actual positive is with nEVD threshold = 0.3
    cm2_entropyConf = {threshold: [[0, 0], [0, 0]] for threshold in entropyConf_thresholds}  # actual positive is with nEVD threshold = 0.2
    cm1_entropyConf = {threshold: [[0, 0], [0, 0]] for threshold in entropyConf_thresholds}  # actual positive is with nEVD threshold = 0.1
    
    entropyConv_per_demo = {i: [] for i in range(0, num_demonstration)}

    max_entropy = np.log(num_steps * (1 - burn_frac))
    logger.info(f"Maximum entropy: {max_entropy:.4f}")
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
        print_policy_2(map_policy, size=size)

        if demonstration == 0:
            mcmc_sample_prior = []
            outer_mcmc_samples_prior = []

        else:
            mcmc_sample_prior = [sample for i in range(1, demonstration+1) for sample in mcmc_samples_history[i]]
            outer_mcmc_samples_prior = mcmc_samples_history[demonstration]

        
        posterior_mcmc_samples = [sample for i in range(1, demonstration+2) for sample in mcmc_samples_history[i]]
        
        print("Len prior samples: ", len(mcmc_sample_prior))
        print("Len posterior samples: ", len(posterior_mcmc_samples))
        
        entropy = compute_entropy(env,
            demos=demos_shuffled[:demonstration+1],
            inner_mcmc_samples_post=posterior_mcmc_samples,
            outer_mcmc_samples_post=mcmc_samples_history[demonstration+1],
            beta=beta,
            log_prob_func=log_prob_demo)
        
        conf_ent = 1 - (entropy / max_entropy)
        #conf_ent = max_entropy - entropy

        entropyConv_per_demo[demonstration].append(conf_ent)
        logger.info(f"Information gain {demonstration + 1} demonstrations: {conf_ent :.6f}")
        
        # evaluate thresholds
        for t in range(len(entropyConf_thresholds)):
            threshold = entropyConf_thresholds[t]
            actual_nevd = calculate_expected_value_difference(eval_policy=map_policy, env=env, epsilon=epsilon, normalize_with_random_policy=random_normalization)
            optimality = calculate_percentage_optimal_actions(map_policy, env)
            if conf_ent >= threshold:
                num_demos_entropyConf[threshold].append(demonstration + 1)
                pct_states_entropyConf[threshold].append((demonstration + 1) / (env.rows * env.columns))
                policy_optimalities_entropyConf[threshold].append(optimality)
                policy_accuracies_entropyConf[threshold].append(calculate_policy_accuracy(policies[i], map_policy))
                accuracies_entropyConf[threshold].append(optimality >= optimality_threshold)
                curr_map_pi = map_policy

                # Evaluate actual positive by optimality
                if optimality >= 1.0:
                    cm100_entropyConf[threshold][0][0] += 1
                else:
                    cm100_entropyConf[threshold][0][1] += 1
                if optimality >= 0.95:
                    cm95_entropyConf[threshold][0][0] += 1
                else:
                    cm95_entropyConf[threshold][0][1] += 1
                if optimality >= 0.90:
                    cm90_entropyConf[threshold][0][0] += 1
                else:
                    cm90_entropyConf[threshold][0][1] += 1
                # Evaluate actual positive by nEVD
                if actual_nevd < 0.5:
                    cm5_entropyConf[threshold][0][0] += 1
                else:
                    cm5_entropyConf[threshold][0][1] += 1
                if actual_nevd < 0.4:
                    cm4_entropyConf[threshold][0][0] += 1
                else:
                    cm4_entropyConf[threshold][0][1] += 1
                if actual_nevd < 0.3:
                    cm3_entropyConf[threshold][0][0] += 1
                else:
                    cm3_entropyConf[threshold][0][1] += 1
                if actual_nevd < 0.2:
                    cm2_entropyConf[threshold][0][0] += 1
                else:
                    cm2_entropyConf[threshold][0][1] += 1
                if actual_nevd < 0.1:
                    cm1_entropyConf[threshold][0][0] += 1
                else:
                    cm1_entropyConf[threshold][0][1] += 1
            else:
                # Evaluate actual positive by optimality
                if optimality >= 1.0:
                    cm100_entropyConf[threshold][1][0] += 1
                else:
                    cm100_entropyConf[threshold][1][1] += 1
                if optimality >= 0.95:
                    cm95_entropyConf[threshold][1][0] += 1
                else:
                    cm95_entropyConf[threshold][1][1] += 1
                if optimality >= 0.90:
                    cm90_entropyConf[threshold][1][0] += 1
                else:
                    cm90_entropyConf[threshold][1][1] += 1
                # Evaluate actual positive by nEVD
                if actual_nevd < 0.5:
                    cm5_entropyConf[threshold][1][0] += 1
                else:
                    cm5_entropyConf[threshold][1][1] += 1
                if actual_nevd < 0.4:
                    cm4_entropyConf[threshold][1][0] += 1
                else:
                    cm4_entropyConf[threshold][1][1] += 1
                if actual_nevd < 0.3:
                    cm3_entropyConf[threshold][1][0] += 1
                else:
                    cm3_entropyConf[threshold][1][1] += 1
                if actual_nevd < 0.2:
                    cm2_entropyConf[threshold][1][0] += 1
                else:
                    cm2_entropyConf[threshold][1][1] += 1
                if actual_nevd < 0.1:
                    cm1_entropyConf[threshold][1][0] += 1
                else:
                    cm1_entropyConf[threshold][1][1] += 1


    ########################################################################
    # Store MCMC samples
    ########################################################################
    mcmc_samples_all_experiments[i + 1] = mcmc_samples_history
    

    
    ########################################################################
    # Store results for entropyConf
    ########################################################################
    num_demos_entropyConf_all.append(num_demos_entropyConf)
    pct_states_entropyConf_all.append(pct_states_entropyConf)
    policy_optimalities_entropyConf_all.append(policy_optimalities_entropyConf)
    policy_accuracies_entropyConf_all.append(policy_accuracies_entropyConf)
    accuracies_entropyConf_all.append(accuracies_entropyConf)
    cm100_entropyConf_all.append(cm100_entropyConf)
    cm95_entropyConf_all.append(cm95_entropyConf)
    cm90_entropyConf_all.append(cm90_entropyConf)
    cm5_entropyConf_all.append(cm5_entropyConf)
    cm4_entropyConf_all.append(cm4_entropyConf)
    cm3_entropyConf_all.append(cm3_entropyConf)
    cm2_entropyConf_all.append(cm2_entropyConf)
    cm1_entropyConf_all.append(cm1_entropyConf)    
    entropyConv_all_experiments.append(entropyConv_per_demo)

    if (i+1)%2 == 0:
        # Save results to files
        logger.info("\nSaving results to files...")
        
        np.save(os.path.join(save_dir, 'num_demos_entropyConf_all.npy'), num_demos_entropyConf_all)
        np.save(os.path.join(save_dir, 'pct_states_entropyConf_all.npy'), pct_states_entropyConf_all)
        np.save(os.path.join(save_dir, 'policy_optimalities_entropyConf_all.npy'), policy_optimalities_entropyConf_all)
        np.save(os.path.join(save_dir, 'policy_accuracies_entropyConf_all.npy'), policy_accuracies_entropyConf_all)
        np.save(os.path.join(save_dir, 'accuracies_entropyConf_all.npy'), accuracies_entropyConf_all)
        np.save(os.path.join(save_dir, 'cm100_entropyConf_all.npy'), cm100_entropyConf_all)
        np.save(os.path.join(save_dir, 'cm95_entropyConf_all.npy'), cm95_entropyConf_all)
        np.save(os.path.join(save_dir, 'cm90_entropyConf_all.npy'), cm90_entropyConf_all)
        np.save(os.path.join(save_dir, 'cm5_entropyConf_all.npy'), cm5_entropyConf_all)
        np.save(os.path.join(save_dir, 'cm4_entropyConf_all.npy'), cm4_entropyConf_all)
        np.save(os.path.join(save_dir, 'cm3_entropyConf_all.npy'), cm3_entropyConf_all)
        np.save(os.path.join(save_dir, 'cm2_entropyConf_all.npy'), cm2_entropyConf_all)
        np.save(os.path.join(save_dir, 'cm1_entropyConf_all.npy'), cm1_entropyConf_all)
        np.save(os.path.join(save_dir, 'entropyConv_all_experiments.npy'), entropyConv_all_experiments)
                
        logger.info("Results saved successfully.")