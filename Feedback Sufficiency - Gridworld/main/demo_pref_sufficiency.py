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
from reward_learning.birl_with_preference import CombinedPreferenceBIRL
from utils.common_helper import (calculate_percentage_optimal_actions,
                                 compute_policy_loss_avar_bound,
                                 calculate_expected_value_difference,)
from utils.env_helper import print_policy

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='Experiment Settings')
parser.add_argument('--num_demonstration', type=int, help='Number of demonstrations', required=False)
parser.add_argument('--num_preference', type=int, help='Number of preferences', required=False)
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
num_demonstration = args.num_demonstration
num_preference = args.num_preference

logger.info(f"Running experiment with {num_world} worlds and {num_demonstration} demonstrations and {num_preference} preferences per world.")

# Initialize environments
# Define your feature weights list
feature_weights_list = [
    [-0.6013717323163695, -0.4011769838940959, -0.039571567953142074, 0.0007725477552799112, 1.1912556489120052],
    [-1.1217980413374955, -0.3285427929909659, -0.046151949077585644, 0.0894146097494482, 1.4711108625382643],
    [-0.7744991674213827, -0.7252371406854163, -0.5756375624122959, 0.6981151210530144, 0.9743012090609982],
    [-1.7023756462011976, -0.6839227446123147, 0.19312307756288158, 0.7883232657932492, 0.9369633242949073],
    [-1.279979055180863, 0.09858000410784414, 0.48838414140677544, 0.8605988979806996, 0.9445281996461155],
    [-0.40048256661443304, 0.4935490477530504, 0.9220240474166675, 1.9001142431958848, 2.030074256608167],
    [-1.0384777863855037, -0.4940236964969169, -0.4233056444203063, 0.29609399150097987, 0.9136814928505771],
    [0.13107450991094663, 0.5775192086353814, 0.7363485678282284, 0.9015598410888027, 1.1023404699759505],
    [-1.8920213491866318, -0.9180998799986702, -0.8223161346726733, 0.19231506507636129, 1.363883158215588],
    [-0.2674808331105525, 0.5463337124124177, 0.554224136778753, 0.6534267074528403, 1.1088191691365583],
    [-1.4907490790199256, -0.7007654688566178, -0.48817307677891564, 0.6027922876684342, 0.6237424848490468],
    [-0.711219243749891, 0.44325370800407415, 0.9509525463209803, 1.364158367964643, 1.42542655314575],
    [-1.548010964374959, -0.7545384190866727, -0.6353205693586598, -0.12312694921876237, 0.4465727737765242],
    [-0.15109831429724777, -0.037086712593335264, 0.3265660346387209, 0.9020456043480123, 1.1341856488358757],
    [-0.829311792916716, -0.06927809257707368, -0.0012492366678955229, 0.3633735175242521, 1.4703519541952081],
    [-0.057538375626338865, 0.06582925652189431, 0.422268658619844, 1.6993054866741903, 2.5860729523019357],
    [-1.090285405589971, 0.1212472160424648, 0.1934779320513319, 1.7927904484464716, 2.388718878604693],
    [-0.5882043626256361, -0.3130237149599682, 0.3253296330678707, 0.9620305927670486, 0.9835195639630158],
    [-0.7883789874476681, 0.1777351221376334, 0.32576413708914514, 0.6064620541039307, 1.4810317057513096],
    [-0.7601203375850313, -0.28377146001762377, -0.25136118116848755, 0.2387799142742502, 1.361316868970312]
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


# Begin world iteration
for i in range(num_world):
    logger.info(f"\nRunning world {i+1}/{num_world}")
    env = envs[i]
    
    avar_bounds = {k: {i: [] for i in range(0, num_demonstration)} for k in alphas}
    true_avar_bounds = {i: [] for i in range(0, num_demonstration)}
    
    bounds = {threshold: [] for threshold in thresholds}
    num_demos = {threshold: [] for threshold in thresholds}
    true_evds = {threshold: [] for threshold in thresholds}
    avg_bound_errors = {threshold: [] for threshold in thresholds}
    policy_optimalities = {threshold: [] for threshold in thresholds}
    confusion_matrices = {threshold: [[0, 0], [0, 0]] for threshold in thresholds}  # predicted by true

