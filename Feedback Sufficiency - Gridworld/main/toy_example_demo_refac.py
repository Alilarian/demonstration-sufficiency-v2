"""
toy_example_demo_all_criterion.py

A refactored script that runs multiple demonstration sufficiency criteria 
(nEVD, Convergence, Held-Out, and InfoGain) in a modular and unified way.
Baseline processing is now separated into functions.
Usage:
  python toy_example_demo_all_criterion.py --num_demonstration 5 --beta 10.0 --save_dir run1
"""

import os
import sys
import random
import copy
import math
import argparse
import logging
import yaml
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp

# Set up import paths
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# Local module imports
from env import gridworld_env2
from agent.q_learning_agent import ValueIteration
from reward_learning.birl import BIRL
from utils.common_helper import (
    calculate_percentage_optimal_actions,
    compute_policy_loss_avar_bound,
    calculate_expected_value_difference,
    calculate_policy_accuracy,
    compute_infogain
)
from utils.env_helper import print_policy

###############################################################################
# A. HELPER FUNCTIONS FOR METRIC INITIALIZATION AND UPDATES
###############################################################################

def init_metrics_nEVD(thresholds):
    return {
        "bounds": {th: [] for th in thresholds},
        "num_demos": {th: [] for th in thresholds},
        "true_evds": {th: [] for th in thresholds},
        "avg_bound_errors": {th: [] for th in thresholds},
        "policy_optimalities": {th: [] for th in thresholds},
        "confusion_matrices": {th: [[0, 0], [0, 0]] for th in thresholds}
    }

def init_metrics_convergence(conv_thresholds):
    return {
        "num_demos_conv": {cth: [] for cth in conv_thresholds},
        "pct_states_conv": {cth: [] for cth in conv_thresholds},
        "policy_optimalities_conv": {cth: [] for cth in conv_thresholds},
        "policy_accuracies_conv": {cth: [] for cth in conv_thresholds},
        "accuracies_conv": {cth: [] for cth in conv_thresholds},
        "cm100_conv": {cth: [[0, 0], [0, 0]] for cth in conv_thresholds},
        "cm95_conv": {cth: [[0, 0], [0, 0]] for cth in conv_thresholds},
        "cm90_conv": {cth: [[0, 0], [0, 0]] for cth in conv_thresholds},
        "cm5_conv":  {cth: [[0, 0], [0, 0]] for cth in conv_thresholds},
        "cm4_conv":  {cth: [[0, 0], [0, 0]] for cth in conv_thresholds},
        "cm3_conv":  {cth: [[0, 0], [0, 0]] for cth in conv_thresholds},
        "cm2_conv":  {cth: [[0, 0], [0, 0]] for cth in conv_thresholds},
        "cm1_conv":  {cth: [[0, 0], [0, 0]] for cth in conv_thresholds}
    }

def init_metrics_heldout(held_out_thresholds):
    return {
        "num_demos_val": {th: [] for th in held_out_thresholds},
        "pct_states_val": {th: [] for th in held_out_thresholds},
        "policy_optimalities_val": {th: [] for th in held_out_thresholds},
        "accuracies_val": {th: [] for th in held_out_thresholds},
        "cm100_val": {th: [[0, 0], [0, 0]] for th in held_out_thresholds},
        "cm95_val": {th: [[0, 0], [0, 0]] for th in held_out_thresholds},
        "cm90_val": {th: [[0, 0], [0, 0]] for th in held_out_thresholds},
        "cm5_val":  {th: [[0, 0], [0, 0]] for th in held_out_thresholds},
        "cm4_val":  {th: [[0, 0], [0, 0]] for th in held_out_thresholds},
        "cm3_val":  {th: [[0, 0], [0, 0]] for th in held_out_thresholds},
        "cm2_val":  {th: [[0, 0], [0, 0]] for th in held_out_thresholds},
        "cm1_val":  {th: [[0, 0], [0, 0]] for th in held_out_thresholds}
    }

def init_metrics_infogain():
    return {"info_gain": []}

def update_confusion_matrix(matrix, predicted_positive, actual_positive):
    if predicted_positive and actual_positive:
        matrix[0][0] += 1  # True Positive
    elif predicted_positive and not actual_positive:
        matrix[0][1] += 1  # False Positive
    elif not predicted_positive and actual_positive:
        matrix[1][0] += 1  # False Negative
    else:
        matrix[1][1] += 1  # True Negative

###############################################################################
# B. HELPER FUNCTIONS FOR BASELINE PROCESSING
###############################################################################

def process_nevd_baseline_for_threshold(env, used_demos, mcmc_samples, demo_idx, threshold, epsilon, rn):
    # Compute nEVD-based metrics using helper functions from common_helper.
    avar_bound = compute_policy_loss_avar_bound(mcmc_samples, env, used_demos, rn, alpha=0.95, delta=0.05)
    true_evd = calculate_expected_value_difference(used_demos, env, epsilon=epsilon, normalize_with_random_policy=rn)
    confusion = [[0, 0], [0, 0]]
    predicted_positive = (avar_bound < threshold)
    actual_positive = (true_evd < threshold)
    update_confusion_matrix(confusion, predicted_positive, actual_positive)
    return {
        "bounds": [avar_bound],
        "num_demos": [demo_idx + 1],
        "true_evds": [true_evd],
        "avg_bound_errors": [avar_bound - true_evd],
        "policy_optimalities": [calculate_percentage_optimal_actions(used_demos, env)],
        "confusion_matrix": confusion
    }

def process_nevd_baseline(env, used_demos, mcmc_samples, demo_idx, thresholds, epsilon, rn):
    result = {}
    for th in thresholds:
        result[th] = process_nevd_baseline_for_threshold(env, used_demos, mcmc_samples, demo_idx, th, epsilon, rn)
    return result

def process_conv_baseline(old_policy, new_policy, demo_idx, conv_thresholds, env, epsilon, rn, optimality_threshold):
    results = {}
    for cth in conv_thresholds:
        match = (new_policy == old_policy)
        results[cth] = {
            "num_demos_conv": [demo_idx + 1] if match else [],
            "pct_states_conv": [((demo_idx + 1) / (env.rows * env.columns))] if match else [],
            "policy_optimalities_conv": [calculate_percentage_optimal_actions(new_policy, env)] if match else [],
            "policy_accuracies_conv": [calculate_policy_accuracy(new_policy, env)] if match else [],
            "accuracies_conv": [calculate_percentage_optimal_actions(new_policy, env) >= optimality_threshold] if match else [],
            "cm100_conv": [[1, 0], [0, 0]] if match else [[0, 0], [0, 0]]
        }
    return results

def process_heldout_baseline(env, used_demos, demo_count, held_out_thresholds):
    results = {}
    for th in held_out_thresholds:
        results[th] = {
            "num_demos_val": [demo_count],
            "pct_states_val": [demo_count / (env.rows * env.columns)],
            "policy_optimalities_val": [calculate_percentage_optimal_actions(used_demos, env)],
            "accuracies_val": [calculate_policy_accuracy(used_demos, env)]
        }
    return results

def process_infogain_baseline(env, used_demos, prior_samples, current_samples, beta):
    ig_value = compute_infogain(env, used_demos, prior_samples, current_samples, beta)
    return {"info_gain": [ig_value]}

def process_nevd_metrics(nevd_metrics, nevd_result, thresholds):
    for th in thresholds:
        nevd_metrics["bounds"][th].extend(nevd_result[th]["bounds"])
        nevd_metrics["num_demos"][th].extend(nevd_result[th]["num_demos"])
        nevd_metrics["true_evds"][th].extend(nevd_result[th]["true_evds"])
        nevd_metrics["avg_bound_errors"][th].extend(nevd_result[th]["avg_bound_errors"])
        nevd_metrics["policy_optimalities"][th].extend(nevd_result[th]["policy_optimalities"])
        for row in range(2):
            for col in range(2):
                nevd_metrics["confusion_matrices"][th][row][col] += nevd_result[th]["confusion_matrix"][row][col]

def process_conv_metrics(conv_metrics, conv_result, conv_thresholds):
    for cth in conv_thresholds:
        conv_metrics["num_demos_conv"][cth].extend(conv_result[cth]["num_demos_conv"])
        conv_metrics["pct_states_conv"][cth].extend(conv_result[cth]["pct_states_conv"])
        conv_metrics["policy_optimalities_conv"][cth].extend(conv_result[cth]["policy_optimalities_conv"])
        conv_metrics["policy_accuracies_conv"][cth].extend(conv_result[cth]["policy_accuracies_conv"])
        conv_metrics["accuracies_conv"][cth].extend(conv_result[cth]["accuracies_conv"])
        # Update confusion matrix for one example confusion matrix (cm100_conv)
        for row in range(2):
            for col in range(2):
                conv_metrics["cm100_conv"][cth][row][col] += conv_result[cth]["cm100_conv"][row][col]

def process_heldout_metrics(heldout_metrics, heldout_result, held_out_thresholds):
    for th in held_out_thresholds:
        heldout_metrics["num_demos_val"][th].extend(heldout_result[th]["num_demos_val"])
        heldout_metrics["pct_states_val"][th].extend(heldout_result[th]["pct_states_val"])
        heldout_metrics["policy_optimalities_val"][th].extend(heldout_result[th]["policy_optimalities_val"])
        heldout_metrics["accuracies_val"][th].extend(heldout_result[th]["accuracies_val"])

###############################################################################
# C. CONFIGURATION MANAGEMENT
###############################################################################

def load_config(config_filename="gridworld_config.yaml"):
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    config_path = os.path.join(parent_dir, "configs", config_filename)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config_params = {
        "gamma": config['env_config']['gamma'],
        "epsilon": float(config['algorithm_config']['epsilon']),
        "num_steps": config['bayesian_irl_config']['num_steps'],
        "step_stdev": config['bayesian_irl_config']['step_stdev'],
        "burn_frac": config['bayesian_irl_config']['burn_frac'],
        "skip_rate": config['bayesian_irl_config']['skip_rate'],
        "adaptive": config['bayesian_irl_config']['adaptive'],
        "thresholds": config['suff_config']['thresholds'],
        "conv_thresholds": config['suff_config']['conv_thresholds'],
        "held_out_thresholds": config['suff_config']['held_out_thresholds'],
        "normalized_infogain_thresholds": config['suff_config'].get('normalized_infogain_thresholds', []),
        "entropy_confidence_thresholds": config['suff_config'].get('entropy_confidence_thresholds', []),
        "alphas": config['suff_config']['alphas'],
        "delta": config['suff_config']['delta'],
        "optimality_threshold": config['suff_config']['optimality_threshold'],
        "random_normalization": config['suff_config']['random_normalization'],
        "num_world": config['experiments']['num_world'],
        "num_demonstration": config['experiments']['num_demonstration']
    }
    return config_params

###############################################################################
# D. MAIN EXPERIMENT FUNCTION
###############################################################################

def run_mcmc_and_get_map(env, demos, beta, num_steps, step_stdev, burn_frac, skip_rate, adaptive):
    birl = BIRL(env, demos, beta)
    birl.run_mcmc(num_steps, step_stdev, adaptive=adaptive)
    chain = birl.chain
    burn_indx = int(len(chain) * burn_frac)
    mcmc_samples = chain[burn_indx::skip_rate]
    map_solution = birl.get_map_solution()
    return map_solution, mcmc_samples, birl

def check_convergence(map_policy, old_map_policy, patience, threshold):
    same = (map_policy == old_map_policy)
    if same:
        patience += 1
    else:
        patience = 0
    has_stopped = (patience >= threshold)
    return patience, has_stopped

def main():
    # 1) Parse command-line arguments
    parser = argparse.ArgumentParser(description='Experiment Settings')
    parser.add_argument('--num_demonstration', type=int, required=True)
    parser.add_argument('--beta', type=float, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    args = parser.parse_args()

    # 2) Setup logging and load config
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    save_dir = os.path.join(parent_dir, "results", args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "default_log.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger()
    
    config = load_config()
    gamma = config["gamma"]
    epsilon = config["epsilon"]
    num_steps = config["num_steps"]
    step_stdev = config["step_stdev"]
    burn_frac = config["burn_frac"]
    skip_rate = config["skip_rate"]
    adaptive = config["adaptive"]
    thresholds = config["thresholds"]
    conv_thresholds = config["conv_thresholds"]
    held_out_thresholds = config["held_out_thresholds"]
    alphas = config["alphas"]
    delta = config["delta"]
    optimality_threshold = config["optimality_threshold"]
    random_normalization = config["random_normalization"]
    num_world = config["num_world"]
    num_demonstration = args.num_demonstration
    beta = float(args.beta)

    # 3) Prepare environments
    color_to_feature_map = {"red": [1, 0, 0], "blue": [0, 1, 0], "black": [0, 0, 1]}
    custom_grid_features = [["blue", "red", "blue"], ["blue", "blue", "black"]]
    feature_weights_list = np.load("grid_world_weights.npy")
    envs = []
    for feat in feature_weights_list:
        env = gridworld_env2.NoisyLinearRewardFeaturizedGridWorldEnv(
            gamma=gamma,
            color_to_feature_map=color_to_feature_map,
            grid_features=custom_grid_features,
            custom_feature_weights=list(feat)
        )
        envs.append(env)
    logger.info(f"Created {len(envs)} environments.")

    policies = [ValueIteration(envs[i]).get_optimal_policy() for i in range(num_world)]
    logger.info("Generated optimal policies for all environments.")

    # For illustration, we use a fixed demonstration set (e.g., pairs)
    demos = [(0, 1), (2, 1), (4, 3)]

    # 4) Initialize overall storage for each baseline
    nevd_data_all = []
    conv_data_all = []
    heldout_data_all = []
    infogain_data_all = []
    mcmc_samples_all_experiments = {}

    # 5) Loop over environment replications
    for env_index in range(num_world):
        env = envs[env_index]
        logger.info(f"\n--- Running environment {env_index+1}/{num_world} ---")
        
        # Initialize per-environment metrics
        nevd_metrics = init_metrics_nEVD(thresholds)
        conv_metrics = init_metrics_convergence(conv_thresholds)
        heldout_metrics = init_metrics_heldout(held_out_thresholds)
        infogain_metrics = init_metrics_infogain()
        
        mcmc_samples_history = {}
        old_map_pi = None
        patience_count = 0
        held_out_sets = {th: [] for th in held_out_thresholds}
        total_demos_heldout = {th: [] for th in held_out_thresholds}
        
        demos_shuffled = demos.copy()
        random.shuffle(demos_shuffled)
        info_gain_results = {i: [] for i in range(num_demonstration)}
        
        # Loop over demonstration iterations
        for d_idx in range(num_demonstration):
            used_demos = demos_shuffled[:d_idx+1]
            map_solution, mcmc_samples, birl_obj = run_mcmc_and_get_map(
                env, used_demos, beta, num_steps, step_stdev, burn_frac, skip_rate, adaptive
            )
            mcmc_samples_history[d_idx+1] = mcmc_samples
            
            map_env = copy.deepcopy(env)
            map_env.set_feature_weights(map_solution)
            map_policy = ValueIteration(map_env).get_optimal_policy()
            logger.info(f"Env {env_index+1}, demo {d_idx+1}: MAP Solution = {map_solution}")
            print_policy(map_policy, 2, 3)
            
            # --- Process nEVD baseline ---
            nevd_result = process_nevd_baseline(env, used_demos, mcmc_samples, d_idx, thresholds, epsilon, random_normalization)
            process_nevd_metrics(nevd_metrics, nevd_result, thresholds)
            
            # --- Process Convergence baseline ---
            if old_map_pi is None:
                old_map_pi = map_policy
            else:
                patience_count, conv_stopped = check_convergence(map_policy, old_map_pi, patience_count, threshold=conv_thresholds[0])
                conv_result = process_conv_baseline(old_map_pi, map_policy, d_idx, conv_thresholds, env, epsilon, random_normalization, optimality_threshold)
                process_conv_metrics(conv_metrics, conv_result, conv_thresholds)
                old_map_pi = map_policy
            
            # --- Process Held-Out baseline ---
            for hth in held_out_thresholds:
                if (d_idx+1) % hth == 0:
                    held_out_sets[hth].append(demos_shuffled[d_idx])
                else:
                    total_demos_heldout[hth].append(demos_shuffled[d_idx])
                    heldout_result = process_heldout_baseline(env, used_demos, d_idx+1, held_out_thresholds)
                    process_heldout_metrics(heldout_metrics, heldout_result, held_out_thresholds)
            
            # --- Process InfoGain baseline ---
            if d_idx == 0:
                prior_samples = []
            else:
                prior_samples = mcmc_samples_history[d_idx]
            ig_result = process_infogain_baseline(env, used_demos, prior_samples, mcmc_samples, beta)
            infogain_metrics["info_gain"].extend(ig_result["info_gain"])
            logger.info(f"Env {env_index+1}, demo {d_idx+1}: InfoGain = {ig_result['info_gain'][0]:.6f}")
            
        # End demonstration loop; store metrics for current environment
        nevd_data_all.append(nevd_metrics)
        conv_data_all.append(conv_metrics)
        heldout_data_all.append(heldout_metrics)
        infogain_data_all.append(infogain_metrics)
        mcmc_samples_all_experiments[env_index+1] = mcmc_samples_history
        logger.info(f"Finished environment {env_index+1}")

    # 6) Save overall results
    np.save(os.path.join(save_dir, 'nevd_data_all.npy'), nevd_data_all)
    np.save(os.path.join(save_dir, 'conv_data_all.npy'), conv_data_all)
    np.save(os.path.join(save_dir, 'heldout_data_all.npy'), heldout_data_all)
    np.save(os.path.join(save_dir, 'infogain_data_all.npy'), infogain_data_all)
    np.save(os.path.join(save_dir, 'mcmc_samples_all_experiments.npy'), mcmc_samples_all_experiments)
    logger.info("All results saved successfully.")

if __name__ == "__main__":
    main()
