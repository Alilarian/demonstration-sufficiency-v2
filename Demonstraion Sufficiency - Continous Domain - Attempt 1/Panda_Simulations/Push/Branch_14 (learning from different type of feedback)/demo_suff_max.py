"""
Repeating experiments with the top 15 instead of 100 to have less variance
"""

import numpy as np
from env_1 import Env1
from algos import *
import math
from scipy.stats import norm

from algos import generate_pairwise_comparisons, PBIRL

def trajectory_optimization(trajectories, traj_feats, reward_param):
    """
    Simulating the planning phase in the trajectory space.

    This function takes a list of the all possible traj_feats and a reward parameter, calculates the reward for each trajectory,
    and returns a sorted list of tuples containing the reward and the trajectory index, sorted by reward in descending order.

    Parameters:
    traj_feats (list of np.ndarray): A list of traj_feats, where each trajectory is represented as a NumPy array of features.
    reward_param (any): The parameter(s) used to calculate the reward for each trajectory.

    Returns:
    list of tuple: A list of tuples where each tuple contains:
                   - curr_reward (float): The calculated reward for the trajectory
                   - traj_index (int): The index of the trajectory in the original list.
                   The list is sorted by the reward in descending order.
    """
    traj_index_with_reward = []  # List of tuples (reward, trajectory index)
    #for traj_index, traj_feature in enumerate(traj_feats):
    for traj, traj_feature in zip(trajectories, traj_feats):

        curr_reward = np.dot(traj_feature, reward_param)

        traj_index_with_reward.append((curr_reward, traj_feature, traj))
    

    #print(traj_index_with_reward)
    return sorted(traj_index_with_reward, key=lambda x: x[0], reverse=True)

#start_position = np.array([0.4, 0.0, 0.25])
#goal_position = np.array([0.75, 0.0, 0.1])
#xi0 = np.linspace(start_position, goal_position, 3)
true_reward_param = np.array([0.92, 0.11, -0.34])

# Define the list of alpha values
alphas = [0.99, 0.95, 0.90, 0.85, 0.75]

n_runs = 200
n_outer_samples = 1000
n_burn = 0.1
delta = 0.05
normalize = False # Normalize the mcmc samples
num_generated_demos = 50000

num_demonstrations = 10
beta = 1

step_size = 0.9

avar_all_iterations_max_norm_map_policy = []
true_avar_all_iterations_max_norm_map_policy = []


avar_all_iterations_max_norm_mean_policy = []
true_avar_all_iterations_max_norm_mean_policy = []

env = Env1(visualize=False)

demonstration_space = np.load('data/demonstration_space.npy', allow_pickle=True)

all_demonstrations = demonstration_space[:15]
f_star = np.load('data/f_star.npy', allow_pickle=True)
xi_star = np.load('data/xi_star.npy', allow_pickle=True)
rewards = np.load('data/rewards.npy', allow_pickle=True)[:15]
trajectory_space = np.load('data/trajectory_space.npy', allow_pickle=True)

max_reward = max(rewards)

#print(rewards.shape)
#ll_demonstrations = np.array(demonstration_space)[:15]
#rewards = np.array(rewards)[:15]
#pairwise_comparisons = generate_pairwise_comparisons(trajs_feats=trajectory_space[:15], trajs_reward=rewards)

print("Mean of reward of top 15 trajectories: ", np.mean(rewards))
print("Std of reward of top 15 trajectories: ", np.std(rewards))

print("Min reward: ", rewards[-1])
print("Max reward: ", rewards[0])

map_trajs = []
mean_trajs = []

acceptance_ratio_per_samples = {i: [] for i in range(0, 10)}

mcmc_samples_all_experiments = [] # for each experiemnt it saves the mcmc samples after observing each demonstration into a dictionary

for experiment_num in range(n_runs):
    print("=================================================  Experiment Panda Push Branch B12.1 %d =================================================" %experiment_num)

    #avar_bounds_random_norm = {k: {i: [] for i in range(1, 16)} for k in alphas}

    #true_avar_bounds_random_norm = {i: [] for i in range(1, 16)}

    avar_bounds_max_norm_map_policy = {k: {i: [] for i in range(1, 11)} for k in alphas}

    true_avar_bounds_max_norm_map_policy = {i: [] for i in range(1, 11)}


    avar_bounds_max_norm_mean_policy = {k: {i: [] for i in range(1, 11)} for k in alphas}

    true_avar_bounds_max_norm_mean_policy = {i: [] for i in range(1, 11)}


    random_indices = np.random.choice(len(all_demonstrations), num_demonstrations, replace=False)
    #trajectory_space = main_trajectory_space[random_indices]
    
    demonstrations = all_demonstrations[random_indices]
    mcmc_samples_per_demo = {}
    for demonstration in range(num_demonstrations):

        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Demo: {demonstration + 1} <<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
        
        print("Running Preference Baysian IRL...")

        pbirl = PBIRL(demonstrations[:demonstration+1], num_features=3, beta=beta, \
                      mcmc_samples=n_outer_samples, step_size=step_size, demo_type='trajectory', normalizing='max', max_reward=max_reward)
        pbirl.run_mcmc()
        all_double_mcmc_sample = pbirl.get_chain()
       
        map_solution = pbirl.get_map_solution()

        accept_ratio = pbirl.accept_ratio
        mean_solution = np.mean(all_double_mcmc_sample, axis=0)
        #mean_solution = map_solution
        # Fetch all mcmc samples
        # fetch map solution
        # acceptance ratio
        # compute mean solution



        #all_double_mcmc_sample, map_solution = mcmc_double(env, demonstrations[:demonstration + 1], n_outer_samples, n_inner_samples, n_burn, len(true_reward_param), beta=beta)
        #all_double_mcmc_sample, map_solution, accept_ratio = mcmc_double_2(env, demonstrations[:demonstration + 1], n_outer_samples, n_inner_samples,\
        #                                                                    n_burn, len(true_reward_param), beta=beta, step_size=step_size, normalize=normalize)
        #mean_solution = np.mean(all_double_mcmc_sample, axis=0)
        
        #if map_solution is None:
        #    map_solution = np.mean(all_double_mcmc_sample, axis=0)
        #    print("MAP solution is not available")
        
        print("Acceptance ration: ", accept_ratio)
        
        mcmc_samples_per_demo[demonstration] = all_double_mcmc_sample
        
        acceptance_ratio_per_samples[demonstration].append(accept_ratio)

        normalized_expected_regrets_max_normalization_ma = []
        nevd_max_norm_map_policy = [] # normalizing expected regret with respect to the max for map policy
        nevd_max_norm_mean_policy = [] # normalizing expected regret with respect to the max for mean policy


        _, best_traj_feat_for_map, best_traj_for_map  = \
            trajectory_optimization(demonstration_space, trajectory_space, reward_param=map_solution)[0]


        _, best_traj_feat_for_mean, best_traj_for_mean = \
            trajectory_optimization(demonstration_space, trajectory_space, reward_param=mean_solution)[0]
        
        map_trajs.append(best_traj_for_map)
        mean_trajs.append(best_traj_for_mean)

        
        for i, mcmc_sample in enumerate(all_double_mcmc_sample):
            best_traj_reward_for_current_sample, best_traj_for_current_sample, _ = \
                trajectory_optimization(demonstration_space, trajectory_space, reward_param=mcmc_sample)[0]
            
            map_policy_reward_under_current_mcmc_sample = env.reward(best_traj_feat_for_map, mcmc_sample)

            mean_policy_reward_under_current_mcmc_sample = env.reward(best_traj_feat_for_mean, mcmc_sample)

            nevd_max_norm_map_policy.append((best_traj_reward_for_current_sample - map_policy_reward_under_current_mcmc_sample))
            nevd_max_norm_mean_policy.append((best_traj_reward_for_current_sample - mean_policy_reward_under_current_mcmc_sample))

        for alpha in alphas:
            N_burned = len(all_double_mcmc_sample)
            k = math.ceil(N_burned * alpha + norm.ppf(1 - delta) * np.sqrt(N_burned * alpha * (1 - alpha)) - 0.5)
            if k >= N_burned:
                k = N_burned - 1
            #normalized_expected_regrets_random_normalization.sort()
            #normalized_expected_regrets_max_normalization.sort()
            nevd_max_norm_map_policy.sort()
            nevd_max_norm_mean_policy.sort()

            avar_bounds_max_norm_map_policy[alpha][demonstration+1].append(nevd_max_norm_map_policy[k])
            avar_bounds_max_norm_mean_policy[alpha][demonstration+1].append(nevd_max_norm_mean_policy[k])


            print(f"{alpha}-VaR-max-normalization for {demonstration + 1} demonstration MAP Policy: {nevd_max_norm_map_policy[k]:.6f}\n")
            print(f"{alpha}-VaR-max-normalization for {demonstration + 1} demonstration Mean Policy: {nevd_max_norm_mean_policy[k]:.6f}\n")

        #map_traj_true_reward, _ = trajectory_optimization(trajectories=trajectory_space_main, reward_param=map_solution)[0]
        map_traj_true_reward = env.reward(best_traj_feat_for_map, true_reward_param)
        mean_traj_true_reward = env.reward(best_traj_feat_for_mean, true_reward_param)
        
        best_traj_true_reward = env.reward(f_star, true_reward_param)

        #map_traj_true_reward = np.sum(calculate_reward_per_step(trajectory_space_main[traj_index_for_map], optimal_reward_function_param))
        # Evaluate random policy under the true reward function
        #random_policy_true_reward = np.sum(calculate_reward_per_step(random_traj, optimal_reward_function_param))

        true_avar_max_norm_map_policy = ((best_traj_true_reward - map_traj_true_reward))
        true_avar_bounds_max_norm_map_policy[demonstration + 1].append(true_avar_max_norm_map_policy)
        print(f"True nEVD for {demonstration + 1} demonstration MAP Policy: {true_avar_max_norm_map_policy:.6f}\n")

        true_avar_max_norm_mean_policy = ((best_traj_true_reward - mean_traj_true_reward))
        true_avar_bounds_max_norm_mean_policy[demonstration + 1].append(true_avar_max_norm_mean_policy)
        print(f"True nEVD for {demonstration + 1} demonstration MEAN Policy: {true_avar_max_norm_mean_policy:.6f}\n")
        

    mcmc_samples_all_experiments.append(mcmc_samples_per_demo)
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

    #if experiment_num%20 == 0:
        #np.save('results/acceptance_ratio_per_samples_B12.1.npy', acceptance_ratio_per_samples)
        #np.save('results/mcmc_samples_B12.1.npy', mcmc_samples_all_experiments)
        #np.save('results/map_trajectories_B12.1.npy', np.array(map_trajs))
        #np.save('results/mean_trajectories_B12.1.npy', np.array(mean_trajs))
        #np.save('results/optimal_trajectory_B12.1.npy', xi_star)
        #np.save('results/avar_all_iterations_max_norm_map_policy_B12.1.npy', np.array(avar_all_iterations_max_norm_map_policy))
        #np.save('results/true_avar_all_iterations_max_norm_map_policy_B12.1.npy', np.array(true_avar_all_iterations_max_norm_map_policy))