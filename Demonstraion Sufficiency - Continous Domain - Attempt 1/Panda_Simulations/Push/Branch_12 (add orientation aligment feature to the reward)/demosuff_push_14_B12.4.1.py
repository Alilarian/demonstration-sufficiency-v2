"""
Repeating experiments with the top 15 instead of 100 to have less variance
"""

import numpy as np
from env_1 import Env1
from algos import *
import math
from scipy.stats import norm
import pickle

def trajectory_optimization(env, trajectories, traj_feats, reward_param):
    """
    Simulating the planning phase in the trajectory space.

    This function takes a list of the all possible traj_feats and a reward parameter, calculates the reward for each trajectory,
    and returns a sorted list of tuples containing the reward and the trajectory index, sorted by reward in descending order.

    Parameters:
    traj_feats (list of np.ndarray): A list of traj_feats, where each trajectory is represented as a NumPy array of features.
    reward_param (any): The parameter(s) used to calculate the reward for each trajectory.

    Returns:
    list of tuple: A list of tuples where each tuple contains:
                   - curr_reward (float): The calculated reward for the trajectory.
                   - traj_index (int): The index of the trajectory in the original list.
                   The list is sorted by the reward in descending order.
    """
    traj_index_with_reward = []  # List of tuples (reward, trajectory index)
    #for traj_index, traj_feature in enumerate(traj_feats):
    for traj, traj_feature in zip(trajectories, traj_feats):
        curr_reward = env.reward(traj_feature, reward_param)
        #traj_index_with_reward.append((curr_reward, traj_index))
        traj_index_with_reward.append((curr_reward, traj_feature, traj))

    return sorted(traj_index_with_reward, key=lambda x: x[0], reverse=True)

start_position = np.array([0.4, 0.0, 0.30])
goal_position = np.array([0.55, 0.5, 0.15])
xi0 = np.linspace(start_position, goal_position, 10)


true_reward_params = []
for i in range(1):   
    v = np.random.randn(3)
    v /= np.linalg.norm(v)
    if v[2] > 0:
        v[2]*=-1

    if v[0] < 0:
        v[0]*=-1
    
    if v[1] < 0:
        v[1]*=-1

    true_reward_params.append(v)

true_reward_params = np.array(true_reward_params)

# Define the list of alpha values
alphas = [0.99, 0.95, 0.90, 0.85, 0.75]

n_runs = len(true_reward_params)
n_inner_samples = 10
n_outer_samples = 30 ## Need to be changed to 100 or 60
n_burn = 0.1
delta = 0.5
normalize = True # Normalize the mcmc samples
num_generated_demos = 10 # Need to increase that to 100000

num_demonstrations = 1 # Need to be changed to 10
beta = 1

step_size = 0.1

avar_all_iterations_max_norm_map_policy = []
true_avar_all_iterations_max_norm_map_policy = []

avar_all_iterations_max_norm_mean_policy = []
true_avar_all_iterations_max_norm_mean_policy = []

all_map_trajs = []
all_mean_trajs = []


results_dict = {}  # To store all experiment results

for experiment_num in range(n_runs):

    iteration_data = {
        "true_weight": true_reward_params[experiment_num],
        "optimal_traj": None,  # will fill in with xi_star
        "map_trajs": [],       # will store the MAP traj after each demo
        "mean_trajs": []       # will store the MEAN traj after each demo
    }

    env = Env1(visualize=False)
    
    demonstration_space, xi_star, trajectory_space, f_star = human_demo_2(
        env, 
        xi0, 
        true_reward_params[experiment_num],
        n_samples=num_generated_demos, 
        n_demos=num_generated_demos
    )

    all_demonstrations = np.array(demonstration_space)[:10]
    trajectory_space = np.array(trajectory_space)

    # Save the "optimal_traj" (xi_star) in your iteration_data dict
    iteration_data["optimal_traj"] = xi_star

    print("=================================================")
    print(f" Experiment {experiment_num} ")
    print("=================================================")
    
    random_indices = np.random.choice(len(all_demonstrations), num_demonstrations, replace=False)
    demonstrations = all_demonstrations[random_indices]

    avar_bounds_max_norm_map_policy = {k: {i: [] for i in range(1, 11)} for k in alphas}
    true_avar_bounds_max_norm_map_policy = {i: [] for i in range(1, 11)}
    avar_bounds_max_norm_mean_policy = {k: {i: [] for i in range(1, 11)} for k in alphas}
    true_avar_bounds_max_norm_mean_policy = {i: [] for i in range(1, 11)}

    for demonstration in range(num_demonstrations):
        print(f">>>>>>>>> Demo: {demonstration + 1} <<<<<<<<<<\n")
        
        print("Running double Metropolis-Hasting...\n")

        # -- MCMC double call --
        all_double_mcmc_sample, map_solution, accept_ratio = mcmc_double_2(
            env, 
            demonstrations[:demonstration + 1], 
            n_outer_samples, 
            n_inner_samples,
            n_burn, 
            len(true_reward_params[experiment_num]), 
            beta=beta, 
            step_size=step_size, 
            normalize=normalize
        )
        mean_solution = np.mean(all_double_mcmc_sample, axis=0)
        if map_solution is None:
            map_solution = mean_solution
        
        print("Acceptance ratio:", accept_ratio)

        #normalized_expected_regrets_max_normalization_ma = []
        nevd_max_norm_map_policy = [] # normalizing expected regret with respect to the max for map policy
        nevd_max_norm_mean_policy = [] # normalizing expected regret with respect to the max for mean policy

        # -- Compute MAP/Mean trajectories --
        _, best_traj_feat_for_map, best_traj_for_map = trajectory_optimization(
            env, demonstration_space, trajectory_space, reward_param=map_solution
        )[0]

        _, best_traj_feat_for_mean, best_traj_for_mean = trajectory_optimization(
            env, demonstration_space, trajectory_space, reward_param=mean_solution
        )[0]
        
        # 2) Append them to the iteration_data dictionary
        iteration_data["map_trajs"].append(best_traj_for_map)
        iteration_data["mean_trajs"].append(best_traj_for_mean)

        for i, mcmc_sample in enumerate(all_double_mcmc_sample):
            best_traj_reward_for_current_sample, best_traj_for_current_sample, _ = \
                trajectory_optimization(env, demonstration_space, trajectory_space, reward_param=mcmc_sample)[0]
            
            map_policy_reward_under_current_mcmc_sample = env.reward(best_traj_feat_for_map, mcmc_sample)

            mean_policy_reward_under_current_mcmc_sample = env.reward(best_traj_feat_for_mean, mcmc_sample)

            nevd_max_norm_map_policy.append((best_traj_reward_for_current_sample - map_policy_reward_under_current_mcmc_sample) / (2))
            nevd_max_norm_mean_policy.append((best_traj_reward_for_current_sample - mean_policy_reward_under_current_mcmc_sample) / (2))

        for alpha in alphas:
            N_burned = len(all_double_mcmc_sample)
            k = math.ceil(N_burned * alpha + norm.ppf(1 - delta) * np.sqrt(N_burned * alpha * (1 - alpha)) - 0.5)
            if k >= N_burned:
                k = N_burned - 1

            nevd_max_norm_map_policy.sort()
            nevd_max_norm_mean_policy.sort()

            avar_bounds_max_norm_map_policy[alpha][demonstration+1].append(nevd_max_norm_map_policy[k])
            avar_bounds_max_norm_mean_policy[alpha][demonstration+1].append(nevd_max_norm_mean_policy[k])


            print(f"{alpha}-VaR-max-normalization for {demonstration + 1} demonstration MAP Policy: {nevd_max_norm_map_policy[k]:.6f}\n")
            print(f"{alpha}-VaR-max-normalization for {demonstration + 1} demonstration Mean Policy: {nevd_max_norm_mean_policy[k]:.6f}\n")


        #map_traj_true_reward, _ = trajectory_optimization(trajectories=trajectory_space_main, reward_param=map_solution)[0]
        map_traj_true_reward = env.reward(best_traj_feat_for_map, true_reward_params[experiment_num])
        mean_traj_true_reward = env.reward(best_traj_feat_for_mean, true_reward_params[experiment_num])
        

        best_traj_true_reward = env.reward(f_star, true_reward_params[experiment_num])

        #map_traj_true_reward = np.sum(calculate_reward_per_step(trajectory_space_main[traj_index_for_map], optimal_reward_function_param))
        # Evaluate random policy under the true reward function
        #random_policy_true_reward = np.sum(calculate_reward_per_step(random_traj, optimal_reward_function_param))

        true_avar_max_norm_map_policy = ((best_traj_true_reward - map_traj_true_reward))
        true_avar_bounds_max_norm_map_policy[demonstration + 1].append(true_avar_max_norm_map_policy)
        print(f"True nEVD for {demonstration + 1} demonstration MAP Policy: {true_avar_max_norm_map_policy:.6f}\n")



        true_avar_max_norm_mean_policy = ((best_traj_true_reward - mean_traj_true_reward))
        true_avar_bounds_max_norm_mean_policy[demonstration + 1].append(true_avar_max_norm_mean_policy)
        print(f"True nEVD for {demonstration + 1} demonstration MEAN Policy: {true_avar_max_norm_mean_policy:.6f}\n")
        
    avar_all_iterations_max_norm_map_policy.append(avar_bounds_max_norm_map_policy)
    true_avar_all_iterations_max_norm_map_policy.append(true_avar_bounds_max_norm_map_policy)

    avar_all_iterations_max_norm_mean_policy.append(avar_bounds_max_norm_mean_policy)
    true_avar_all_iterations_max_norm_mean_policy.append(true_avar_bounds_max_norm_mean_policy)

    mean_results_map = {k: {i: [] for i in range(1, 11)} for k in alphas}

    true_mean_results_map = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}
    
    mean_results_mean = {k: {i: [] for i in range(1, 11)} for k in alphas}

    true_mean_results_mean = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}
    
    for data in avar_all_iterations_max_norm_map_policy:
        for alpha, val1 in data.items():
            for item , val2 in val1.items():
                mean_results_map[alpha][item].append(val2[0])
    
    for data in true_avar_all_iterations_max_norm_map_policy:
        for key, values in data.items():
            true_mean_results_map[key].append(values[0])

    for data in avar_all_iterations_max_norm_mean_policy:
        for alpha, val1 in data.items():
            for item , val2 in val1.items():
                mean_results_mean[alpha][item].append(val2[0])
    
    for data in true_avar_all_iterations_max_norm_mean_policy:
        for key, values in data.items():
            true_mean_results_mean[key].append(values[0])

    for alpha in alphas:
        print(f"\nMean results - MAP Policy -alpha={alpha}:")
        for i in range(1, 11):
            print(f"Demo:{i} Mean nEVD - Normalized by Max {alpha}: {np.nanmean(mean_results_map[alpha][i]):.6f}")

    print(f"\nTrue Mean result - MAP Policy")
    for i in range(1, 11):
        print(f"Demo:{i} True Mean nEVD: {np.nanmean(true_mean_results_map[i]):.6f}")

    for alpha in alphas:
        print(f"\nMean results - MEAN Policy -alpha={alpha}:")
        for i in range(1, 11):
            print(f"Demo:{i} Mean nEVD - Normalized by Max: {np.nanmean(mean_results_mean[alpha][i]):.6f}")

    print(f"\nTrue Mean result - MEAN Policy")
    for i in range(1, 11):
        print(f"Demo:{i} True Mean nEVD: {np.nanmean(true_mean_results_mean[i]):.6f}")

    results_dict[experiment_num] = iteration_data
    
    
    #if experiment_num%1 == 0:
with open("results/final_results_B12.4.1.pkl", "wb") as f:
    pickle.dump(results_dict, f)
