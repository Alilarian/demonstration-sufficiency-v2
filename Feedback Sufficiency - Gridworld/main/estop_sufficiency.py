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

from envs import gridworld_env
from agent.q_learning_agent import ValueIteration
from data_generation.generate_data import GridWorldMDPDataGenerator
from reward_learning.ebirl import EBIRL
from utils.common_helper import (calculate_percentage_optimal_actions,
                                 compute_policy_loss_avar_bound,
                                 calculate_expected_value_difference)
from utils.env_helper import print_policy

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
num_world = config['experiments']['num_world']
num_demonstration = args.num_demonstration if args.num_demonstration else config['experiments']['num_demonstration']

logger.info(f"Running experiment with {num_world} worlds and {num_demonstration} demonstrations per world.")

# Initialize environments
# Define your feature weights list
feature_weights_list = [
[-0.7228937044187956, -0.6039866618031474, 0.716980698837855, 1.410272623044544, 1.9116754497034756, 2.2209327833693218, 2.838008562723217],
[-1.5061238902756908, -1.0587568956026059, -0.5975650155356756, -0.46850302512248, -0.2559663879947577, 0.8252329567608293, 1.0683281996402703],
[-0.28459900833585844, -0.12226180968411103, -0.10124999619840261, -0.0813868704426967, 0.4424891258702808, 0.6655145900777971, 0.7400782198099581],
[-1.6951417620338811, -1.1385137971618406, -1.0164573091672138, -0.8149772041596924, 0.30368296600812406, 0.7273550479687881, 0.9531846595477876],
[-1.163925272630476, -0.6504958044475895, 0.2010952575187768, 0.2947724632011124, 0.4832152228314785, 0.6545849844962435, 1.6178800831100657],
[-1.5886722828551696, -0.5601279322371531, -0.33745488197426055, -0.24385512395984563, 0.5847131605868083, 1.3656957016021474, 1.3890070183145082],
[-0.4246324339962069, -0.4194070000781329, 0.21137717541056272, 0.3597180668697446, 0.9708110103212171, 1.8009517781002586, 2.4308626364182246],
[-1.7515266763018744, -1.022816937219687, -0.7875235853668652, -0.38131454325393627, -0.2359907807581513, -0.0076188936349312125, 0.3930885510295719],
[-2.0494473859249482, -1.2529739383417635, -1.191745251600571, -1.173635662179249, -0.44286531494819126, 0.14925977513348077, 1.3517570437958357],
[-1.2663059209692278, -0.16355580797106226, -0.08448018285321454, 0.9311977667904227, 1.480428811434332, 1.4943811336578663, 1.842971483323242],
[-1.9814035067396374, -1.7739147247436253, -0.30950337314774706, 0.16185599569005263, 0.5971935855978966, 0.7539320095287584, 1.019668894025018],
[-2.389550393534453, -1.2233085169355755, -1.0032110650717947, 0.2096579843735842, 0.32220030610525224, 1.1403637315574635, 1.420420166502473],
[-2.8851314645277912, -0.859299414664153, -0.06707403259237905, 0.386955696760212, 0.560180351344327, 1.213254223171574, 1.2179612005602338],
[-1.6935002095322327, -1.2105415491873401, -0.5419175106347648, -0.05262939655448498, 1.1876125995082725, 1.2059107029243492, 1.2566182804374377],
[-0.9567457155348924, -0.009718831756065444, 0.24619730966058667, 0.3463996956848992, 0.511716022654165, 0.8964076068111253, 1.2078497639751637],
[-1.8251267577295456, 0.04455146268936877, 0.1373730459762746, 0.16755930516572462, 0.2909711777583544, 0.33061258678988664, 0.6771338537238046],
[-1.6595715160732576, -0.5775846924520861, -0.011239907555791195, 0.1346904611615679, 0.2343538897975466, 0.4092601112071007, 0.4891531488205212],
[-2.009324083582754, -1.3842478843705863, -1.0016083962251447, -0.1274119082350621, -0.12694627351976517, -0.04064021299898549, 0.953728475796138],
[-2.387306315622788, -0.10530812984473704, 0.08755228806505208, 0.168636999442987, 0.3994486557696571, 0.6355113899129147, 2.222396931650872],
[-1.9064125343510276, -1.4822777477740723, -1.342019816500841, -0.9859885898721917, -0.14407329257098778, 0.36660673590601933, 0.9048719663230285],
]

# Initialize environments with feature weights
envs = [gridworld_env.NoisyLinearRewardFeaturizedGridWorldEnv(gamma=gamma, size=size, noise_prob=noise_prob) 
        for _ in range(num_world)]

# Loop through each environment and set feature weights
for env, weights in zip(envs, feature_weights_list):
    env.set_feature_weights(weights)

for env in envs:
    logger.info(f"Feature weights for environment: {env.feature_weights}")

logger.info(f"Initialized {num_world} GridWorld environments.")

# Generate policies for each environment
policies = [ValueIteration(envs[i]).get_optimal_policy() for i in range(num_world)]
logger.info(f"Generated optimal policies for all environments.")

# Initialize metrics storage
demos = [[] for _ in range(num_world)]
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

    pairwise_comparisons = GridWorldMDPDataGenerator(env=env, seed=seed).generate_estop(beta=beta, num_trajs=num_demonstration)
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
    for demonstration in range(num_demonstration):
        logger.info(f"\nRunning PBIRL with {demonstration + 1} demonstrations for environment {i+1}")

        birl = EBIRL(env, pairwise_comparisons[:demonstration], beta)
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
        print_policy(map_policy, size)

        # Calculate a-VaR for different alphas
        for alpha in alphas:
            avar_bound = compute_policy_loss_avar_bound(mcmc_samples, env, map_policy, random_normalization, alpha, delta)
            avar_bounds[alpha][demonstration].append(avar_bound)
            logger.info(f"{alpha}-VaR-max-normalization for {demonstration + 1} demonstrations: {avar_bound:.6f}")

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

    if (i+1)%4 == 0:
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