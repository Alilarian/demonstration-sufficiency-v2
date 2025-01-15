"""

"""
import numpy as np
from env_1 import Env1
from algos import *
import math
from scipy.stats import norm

def trajectory_optimization(env, trajectories, reward_param):
    """
    Simulating the planning phase in the trajectory space.

    This function takes a list of the all possible trajectories and a reward parameter, calculates the reward for each trajectory,
    and returns a sorted list of tuples containing the reward and the trajectory index, sorted by reward in descending order.

    Parameters:
    trajectories (list of np.ndarray): A list of trajectories, where each trajectory is represented as a NumPy array of features.
    reward_param (any): The parameter(s) used to calculate the reward for each trajectory.

    Returns:
    list of tuple: A list of tuples where each tuple contains:
                   - curr_reward (float): The calculated reward for the trajectory.
                   - traj_index (int): The index of the trajectory in the original list.
                   The list is sorted by the reward in descending order.
    """
    traj_index_with_reward = []  # List of tuples (reward, trajectory index)
    for traj_index, traj_feature in enumerate(trajectories):
        curr_reward = env.reward(traj_feature, reward_param)
        #traj_index_with_reward.append((curr_reward, traj_index))
        traj_index_with_reward.append((curr_reward, traj_feature))

    return sorted(traj_index_with_reward, key=lambda x: x[0], reverse=True)

start_position = np.array([0.4, 0.0, 0.5])
goal_position = np.array([0.75, 0.0, 0.3])
xi0 = np.linspace(start_position, goal_position, 3)

true_reward_param = np.array([0.99, 0.1])

# Define the list of alpha values
alphas = [0.99, 0.95, 0.90, 0.85, 0.75]

n_runs = 200
n_inner_samples = 10
n_outer_samples = 200
n_burn = 0.1
delta = 0.5
normalize = False # Normalize the mcmc samples
num_generated_demos = 50000

num_demonstrations = 10
beta = 1
step_size = 0.1

avar_all_iterations_max_norm_map_policy = []
true_avar_all_iterations_max_norm_map_policy = []

avar_all_iterations_max_norm_mean_policy = []
true_avar_all_iterations_max_norm_mean_policy = []

env = Env1(visualize=False)

all_demonstrations, xi_star, trajectory_space, f_star, rewards = human_demo_2(env, xi0, true_reward_param,\
                                                                           n_samples=num_generated_demos, n_demos=5000)

all_demonstrations = np.array(all_demonstrations)[:15] # Picking the first 100's optimal trajectories
rewards = np.array(rewards)[:15]
print("Mean of reward of top 100 trajectories: ", np.mean(rewards))
print("Std of reward of top 100 trajectories: ", np.std(rewards))

trajectory_space = np.array(trajectory_space)

map_trajs = {i: [] for i in range(0, 10)}
mean_trajs = {i: [] for i in range(0, 10)}
acceptance_ratio_per_samples = {i: [] for i in range(0, 10)}

for experiment_num in range(n_runs):
    print("=================================================  Experiment Panda Close Branch 5.3.1 %d =================================================" %experiment_num)

    avar_bounds_max_norm_map_policy = {k: {i: [] for i in range(1, 11)} for k in alphas}

    true_avar_bounds_max_norm_map_policy = {i: [] for i in range(1, 11)}

    avar_bounds_max_norm_mean_policy = {k: {i: [] for i in range(1, 11)} for k in alphas}

    true_avar_bounds_max_norm_mean_policy = {i: [] for i in range(1, 11)}

    random_indices = np.random.choice(len(all_demonstrations), num_demonstrations, replace=False)

    demonstrations = all_demonstrations[random_indices]

    for demonstration in range(num_demonstrations):

        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Demo: {demonstration + 1} <<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
        
        print("Running double Metropolis-Hasting...\n")

        #all_double_mcmc_sample, map_solution = mcmc_double(env, demonstrations[:demonstration + 1], n_outer_samples, n_inner_samples, n_burn, len(true_reward_param), beta=beta)
        all_double_mcmc_sample, map_solution, accept_ratio = mcmc_double_2(env, demonstrations[:demonstration + 1], n_outer_samples, n_inner_samples,\
                                                                            n_burn, len(true_reward_param), beta=beta, step_size=step_size, normalize=normalize)
        acceptance_ratio_per_samples[demonstration].append(accept_ratio)

        mean_solution = np.mean(all_double_mcmc_sample, axis=0)
        if map_solution is None:
            map_solution = np.mean(all_double_mcmc_sample, axis=0)
            print("MAP solution is not available")
        
        print("Acceptance ratio: \n", accept_ratio)

        _, best_traj_for_map = \
            trajectory_optimization(env, trajectories=trajectory_space, reward_param=map_solution)[0]

        _, best_traj_for_mean = \
            trajectory_optimization(env, trajectories=trajectory_space, reward_param=mean_solution)[0]

        map_trajs[demonstration].append(best_traj_for_map)
        mean_trajs[demonstration].append(best_traj_for_mean)

        nevd_max_norm_map_policy = [] # normalizing expected regret with respect to the max for map policy
        nevd_max_norm_mean_policy = [] # normalizing expected regret with respect to the max for mean policy

        for i, mcmc_sample in enumerate(all_double_mcmc_sample):
            best_traj_reward_for_current_sample, best_traj_for_current_sample = trajectory_optimization(env, trajectories=trajectory_space, reward_param=mcmc_sample)[0]
            
            map_policy_reward_under_current_mcmc_sample = env.reward(best_traj_for_map, mcmc_sample)

            mean_policy_reward_under_current_mcmc_sample = env.reward(best_traj_for_mean, mcmc_sample)

            nevd_max_norm_map_policy.append((best_traj_reward_for_current_sample - map_policy_reward_under_current_mcmc_sample))
            nevd_max_norm_mean_policy.append((best_traj_reward_for_current_sample - mean_policy_reward_under_current_mcmc_sample))

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

        map_traj_true_reward = env.reward(best_traj_for_map, true_reward_param)
        mean_traj_true_reward = env.reward(best_traj_for_mean, true_reward_param)
        

        best_traj_true_reward = env.reward(f_star, true_reward_param)

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
            print(f"Demo:{i} Mean nEVD {alpha}: {np.nanmean(mean_results_map[alpha][i]):.6f}")

    print(f"\nTrue Mean result - MAP Policy")
    for i in range(1, 11):
        print(f"Demo:{i} True Mean nEVD: {np.nanmean(true_mean_results_map[i]):.6f}")

    for alpha in alphas:
        print(f"\nMean results - MEAN Policy -alpha={alpha}:")
        for i in range(1, 11):
            print(f"Demo:{i} Mean nEVD: {np.nanmean(mean_results_mean[alpha][i]):.6f}")

    print(f"\nTrue Mean result - MEAN Policy")
    for i in range(1, 11):
        print(f"Demo:{i} True Mean nEVD: {np.nanmean(true_mean_results_mean[i]):.6f}")

    if experiment_num%20 == 0:
        np.save('results/resultmap_trajectories_B5.3.1.npy', np.array(map_trajs))
        np.save('results/mean_trajectories_B5.3.1.npy', np.array(mean_trajs))
        np.save('results/optimal_trajectory_B5.3.1.npy', xi_star)
        np.save('results/avar_all_iterations_max_norm_map_policy_B5.3.1.npy', np.array(avar_all_iterations_max_norm_map_policy))
        np.save('results/true_avar_all_iterations_max_norm_map_policy_B5.3.1.npy', np.array(true_avar_all_iterations_max_norm_map_policy))