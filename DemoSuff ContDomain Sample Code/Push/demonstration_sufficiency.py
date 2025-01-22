"""
This module demonstrates the implementation and main procedure of a demonstration sufficiency algorithm.
It illustrates how to:
1. Generate demonstrations using a human demonstration function (`human_demo_2`).
2. Perform double Metropolis-Hastings MCMC sampling (`mcmc_double_2`) to infer reward parameters.
3. Use those inferred reward parameters (both MAP and mean solutions) to optimize trajectories.
4. Calculate and track nEVD (normalized Expected Value of Deviation) metrics for the MAP and mean solutions.
5. Compare results against the "true" reward parameters.
"""

import numpy as np
from env_1 import Env1   # Custom environment class (assume it provides env.reward, etc.)
from algos import *      # Custom algorithms (assumed to contain human_demo_2, mcmc_double_2, etc.)
import math
from scipy.stats import norm
from utils import trajectory_optimization

# ---------------------
# Initialization
# ---------------------

# Define start/goal positions and initial "straight-line" demonstration
start_position = np.array([0.4, 0.0, 0.25])
goal_position = np.array([0.75, 0.0, 0.1])
xi0 = np.linspace(start_position, goal_position, 3)

# Define the "true" reward parameters for the environment
true_reward_param = np.array([1, 0.1, -0.05])

# List of confidence levels for CVaR/AVaR
alphas = [0.99, 0.95, 0.90, 0.85, 0.75]

# Experiment hyperparameters
n_runs = 300            # Number of experiment repetitions
n_inner_samples = 10    # Inner MCMC iterations
n_outer_samples = 60    # Outer MCMC iterations
n_burn = 0.1            # Fraction of samples to discard (burn-in)
delta = 0.5             # Parameter for VaR boundary
normalize = False       # Whether to normalize MCMC samples
num_generated_demos = 5000  # Total demonstrations to generate

num_demonstrations = 10 # How many demonstrations we will randomly select to learn from
beta = 1                # Parameter controlling (possibly) MCMC acceptance ratio
step_size = 0.5         # Step size for the Metropolis-Hastings proposal distribution

# Data structures to store results across experiments
avar_all_iterations_max_norm_map_policy = []
true_avar_all_iterations_max_norm_map_policy = []
avar_all_iterations_max_norm_mean_policy = []
true_avar_all_iterations_max_norm_mean_policy = []

# Create environment instance (assumes Env1 has a constructor that can take visualize=False)
env = Env1(visualize=False)

# Generate a large set of demonstrations and the full trajectory space
# human_demo_2 is assumed to return:
#   - all_demonstrations: list of demonstration trajectories
#   - xi_star: some reference trajectory (not always used here)
#   - main_trajectory_space: the full set of possible trajectories
#   - f_star: the feature representing the trajectory that is genuinely optimal under true_reward_param
all_demonstrations, xi_star, main_trajectory_space, f_star = human_demo_2(
    env,
    xi0,
    true_reward_param,
    n_samples=num_generated_demos,
    n_demos=1000
)

# Convert to NumPy arrays for easier manipulation
all_demonstrations = np.array(all_demonstrations)
main_trajectory_space = np.array(main_trajectory_space)

print("# requested demos: %f vs # generated demos: %f" % (num_generated_demos, len(all_demonstrations)))

# ------------------------------------------------------------------
# Main Loop: Repeat over several runs (each run is a an independent experiment)
# ------------------------------------------------------------------

for experiment_num in range(n_runs):
    print("=================================================  Experiment Panda Push Branch 3.0.4 %d =================================================" 
          % experiment_num)

    # We store results for each demonstration in dictionaries keyed by alpha and the demonstration index
    avar_bounds_max_norm_map_policy = {k: {i: [] for i in range(1, 11)} for k in alphas}
    true_avar_bounds_max_norm_map_policy = {i: [] for i in range(1, 11)}

    avar_bounds_max_norm_mean_policy = {k: {i: [] for i in range(1, 11)} for k in alphas}
    true_avar_bounds_max_norm_mean_policy = {i: [] for i in range(1, 11)}

    # Randomly pick a subset of demonstrations for this run
    random_indices = np.random.choice(len(all_demonstrations), num_demonstrations, replace=False)
    trajectory_space = main_trajectory_space[random_indices]
    demonstrations = all_demonstrations[random_indices]

    # -----------------------------------------------------------
    #  Loop over the number of demonstrations (incrementally)
    # -----------------------------------------------------------
    for demonstration in range(num_demonstrations):
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Demo: {demonstration + 1} <<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
        print("Running double Metropolis-Hasting...\n")

        # Perform double Metropolis-Hastings sampling
        # mcmc_double_2 is assumed to return:
        #   - all_double_mcmc_sample: all accepted samples from the MCMC
        #   - map_solution: the MAP solution from the chain
        #   - accept_ratio: acceptance ratio for MCMC
        all_double_mcmc_sample, map_solution, accept_ratio = mcmc_double_2(
            env,
            demonstrations[:demonstration + 1],
            n_outer_samples,
            n_inner_samples,
            n_burn,
            len(true_reward_param),
            beta=beta,
            step_size=step_size,
            normalize=normalize
        )

        # Compute mean solution from the MCMC samples
        mean_solution = np.mean(all_double_mcmc_sample, axis=0)
        
        if map_solution is None:
            # Fallback if MAP solution is not available
            map_solution = mean_solution
            print("MAP solution is not available")

        print("Acceptance ratio: \n", accept_ratio)

        # Lists for storing normalized expected regrets
        nevd_max_norm_map_policy = []   # for MAP policy
        nevd_max_norm_mean_policy = []  # for MEAN policy

        # 1) Find best trajectory under the MAP solution
        _, best_traj_for_map = trajectory_optimization(
            env,
            trajectories=trajectory_space,
            reward_param=map_solution
        )[1]

        # 2) Find best trajectory under the MEAN solution
        _, best_traj_for_mean = trajectory_optimization(
            env,
            trajectories=trajectory_space,
            reward_param=mean_solution
        )[1]

        # -------------------------------------------------------
        #  Evaluate performance under each sample from the chain
        # -------------------------------------------------------
        for i, mcmc_sample in enumerate(all_double_mcmc_sample):
            # Identify the best trajectory for this sample
            best_traj_reward_for_current_sample, _ = trajectory_optimization(
                env,
                trajectories=trajectory_space,
                reward_param=mcmc_sample
            )[0]
            
            # Evaluate MAP policy under current sample
            map_policy_reward_under_current_mcmc_sample = env.reward(best_traj_for_map, mcmc_sample)

            # Evaluate MEAN policy under current sample
            mean_policy_reward_under_current_mcmc_sample = env.reward(best_traj_for_mean, mcmc_sample)

            # Here, the difference between best trajectory reward and policy reward is normalized by 2
            # (which presumably is a domain-dependent maximum difference).
            nevd_max_norm_map_policy.append(
                (best_traj_reward_for_current_sample - map_policy_reward_under_current_mcmc_sample) / 2
            )
            nevd_max_norm_mean_policy.append(
                (best_traj_reward_for_current_sample - mean_policy_reward_under_current_mcmc_sample) / 2
            )

        # -------------------------------------
        #  Compute alpha-VaR for the nEVD lists
        # -------------------------------------
        for alpha in alphas:
            N_burned = len(all_double_mcmc_sample)
            # Compute the index k for the alpha-VaR with delta confidence band
            k = math.ceil(
                N_burned * alpha
                + norm.ppf(1 - delta) * np.sqrt(N_burned * alpha * (1 - alpha))
                - 0.5
            )
            if k >= N_burned:
                k = N_burned - 1

            # Sort the arrays to find the VaR
            nevd_max_norm_map_policy.sort()
            nevd_max_norm_mean_policy.sort()

            # Store the alpha-VaR values for this demonstration count
            avar_bounds_max_norm_map_policy[alpha][demonstration + 1].append(nevd_max_norm_map_policy[k])
            avar_bounds_max_norm_mean_policy[alpha][demonstration + 1].append(nevd_max_norm_mean_policy[k])

            print(f"{alpha}-VaR-max-normalization for {demonstration + 1} demonstration MAP Policy: {nevd_max_norm_map_policy[k]:.6f}\n")
            print(f"{alpha}-VaR-max-normalization for {demonstration + 1} demonstration Mean Policy: {nevd_max_norm_mean_policy[k]:.6f}\n")

        # --------------------------------------
        #  Compute true nEVD under the true reward
        # --------------------------------------
        map_traj_true_reward = env.reward(best_traj_for_map, true_reward_param)
        mean_traj_true_reward = env.reward(best_traj_for_mean, true_reward_param)
        
        # Evaluate the truly optimal trajectory's reward
        best_traj_true_reward = env.reward(f_star, true_reward_param)

        # MAP policy's true nEVD
        true_avar_max_norm_map_policy = (best_traj_true_reward - map_traj_true_reward) / 2
        true_avar_bounds_max_norm_map_policy[demonstration + 1].append(true_avar_max_norm_map_policy)
        print(f"True nEVD for {demonstration + 1} demonstration MAP Policy: {true_avar_max_norm_map_policy:.6f}\n")

        # Mean policy's true nEVD
        true_avar_max_norm_mean_policy = (best_traj_true_reward - mean_traj_true_reward) / 2
        true_avar_bounds_max_norm_mean_policy[demonstration + 1].append(true_avar_max_norm_mean_policy)
        print(f"True nEVD for {demonstration + 1} demonstration MEAN Policy: {true_avar_max_norm_mean_policy:.6f}\n")
        
    # -------------------------------------------------------
    # After finishing all demonstrations for this experiment,
    # we append the result dictionaries to global lists.
    # -------------------------------------------------------
    avar_all_iterations_max_norm_map_policy.append(avar_bounds_max_norm_map_policy)
    true_avar_all_iterations_max_norm_map_policy.append(true_avar_bounds_max_norm_map_policy)

    avar_all_iterations_max_norm_mean_policy.append(avar_bounds_max_norm_mean_policy)
    true_avar_all_iterations_max_norm_mean_policy.append(true_avar_bounds_max_norm_mean_policy)

    # Prepare to compute and print mean results so far
    mean_results_map = {k: {i: [] for i in range(1, 11)} for k in alphas}
    true_mean_results_map = {i: [] for i in range(1, 11)}
    
    mean_results_mean = {k: {i: [] for i in range(1, 11)} for k in alphas}
    true_mean_results_mean = {i: [] for i in range(1, 11)}
    
    # Collect MAP-based results
    for data in avar_all_iterations_max_norm_map_policy:
        for alpha, val1 in data.items():
            for item, val2 in val1.items():
                mean_results_map[alpha][item].append(val2[0])
    
    # Collect MAP-based true results
    for data in true_avar_all_iterations_max_norm_map_policy:
        for key, values in data.items():
            true_mean_results_map[key].append(values[0])

    # Collect MEAN-based results
    for data in avar_all_iterations_max_norm_mean_policy:
        for alpha, val1 in data.items():
            for item, val2 in val1.items():
                mean_results_mean[alpha][item].append(val2[0])
    
    # Collect MEAN-based true results
    for data in true_avar_all_iterations_max_norm_mean_policy:
        for key, values in data.items():
            true_mean_results_mean[key].append(values[0])

    # -------------------------------
    # Print out the aggregated stats
    # -------------------------------
    for alpha in alphas:
        print(f"\nMean results - MAP Policy - alpha={alpha}:")
        for i in range(1, 11):
            print(f"Demo:{i} Mean nEVD - Normalized by Max {alpha}: {np.nanmean(mean_results_map[alpha][i]):.6f}")

    print(f"\nTrue Mean result - MAP Policy")
    for i in range(1, 11):
        print(f"Demo:{i} True Mean nEVD: {np.nanmean(true_mean_results_map[i]):.6f}")

    for alpha in alphas:
        print(f"\nMean results - MEAN Policy - alpha={alpha}:")
        for i in range(1, 11):
            print(f"Demo:{i} Mean nEVD - Normalized by Max: {np.nanmean(mean_results_mean[alpha][i]):.6f}")

    print(f"\nTrue Mean result - MEAN Policy")
    for i in range(1, 11):
        print(f"Demo:{i} True Mean nEVD: {np.nanmean(true_mean_results_mean[i]):.6f}")
