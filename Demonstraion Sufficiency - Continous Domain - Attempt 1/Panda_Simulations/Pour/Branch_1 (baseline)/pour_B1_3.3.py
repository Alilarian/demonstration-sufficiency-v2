import numpy as np
from env_1 import Env1
from algos import *
import math
from scipy.stats import norm
from utils import trajectory_optimization

#trajectory parameters
start_position = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
goal_position = np.array([0.288, 0.128, 0.159, -1.752, -2.256, 2.736, -0.096])
xi0 = np.linspace(start_position, goal_position, 4)

true_reward_param = np.array([0.94127807, 0.33763235])

# Define the list of alpha values
alphas = [0.99, 0.95, 0.90, 0.85, 0.75]

n_runs = 200
n_inner_samples = 10
n_outer_samples = 200
n_burn = 0.1
delta = 0.3
normalize = False # Normalize the mcmc samples
num_generated_demos = 50000

num_demonstrations = 10
beta = 1

step_size = 0.3

avar_all_iterations_max_norm_map_policy = []
true_avar_all_iterations_max_norm_map_policy = []


avar_all_iterations_max_norm_mean_policy = []
true_avar_all_iterations_max_norm_mean_policy = []

env = Env1(visualize=False)

trajectory_space = np.load('data/trajectory_space_len4.npy', allow_pickle=True)
demonstration_space = np.load('data/demonstration_space_len4.npy', allow_pickle=True)
xi_star = np.load('data/xi_star_len4.npy', allow_pickle=True)
f_star = np.load('data/f_star_len4.npy', allow_pickle=True)
rewards = np.load('data/rewards_len4.npy', allow_pickle=True)


all_demonstrations = np.array(demonstration_space)[:100]
rewards = np.array(rewards)[:100]
trajectory_space = np.array(trajectory_space)

print("Mean of reward of top 100 trajectories: ", np.mean(rewards))
print("Std of reward of top 100 trajectories: ", np.std(rewards))

print("Min reward: ", rewards[-1])
print("Max reward: ", rewards[0])

map_trajs = []
mean_trajs = []

acceptance_ratio_per_samples = {i: [] for i in range(0, 10)}
mcmc_samples_all_experiments = [] # for each experiemnt it saves the mcmc samples after observing each demonstration into a dictionary

for experiment_num in range(n_runs):
    print("=================================================  Experiment Panda Pour B1_3.3 %d =================================================" %experiment_num)

    avar_bounds_max_norm_map_policy = {k: {i: [] for i in range(1, 11)} for k in alphas}
    true_avar_bounds_max_norm_map_policy = {i: [] for i in range(1, 11)}

    avar_bounds_max_norm_mean_policy = {k: {i: [] for i in range(1, 11)} for k in alphas}
    true_avar_bounds_max_norm_mean_policy = {i: [] for i in range(1, 11)}

    random_indices = np.random.choice(len(all_demonstrations), num_demonstrations, replace=False)

    demonstrations = all_demonstrations[random_indices]
    mcmc_samples_per_demo = {}
    for demonstration in range(num_demonstrations):

        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Demo: {demonstration + 1} <<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
        
        print("Running double Metropolis-Hasting...\n")

        #all_double_mcmc_sample, map_solution = mcmc_double(env, demonstrations[:demonstration + 1], n_outer_samples, n_inner_samples, n_burn, len(true_reward_param), beta=beta)
        all_double_mcmc_sample, map_solution, accept_ratio = mcmc_double_2(env, demonstrations[:demonstration + 1], n_outer_samples, n_inner_samples,\
                                                                            n_burn, len(true_reward_param), beta=beta, step_size=step_size, normalize=normalize)
        mean_solution = np.mean(all_double_mcmc_sample, axis=0)
        
        if map_solution is None:
            map_solution = np.mean(all_double_mcmc_sample, axis=0)
            print("MAP solution is not available")
        
        print("Acceptance ration: ", accept_ratio)
        
        mcmc_samples_per_demo[demonstration] = all_double_mcmc_sample
        
        acceptance_ratio_per_samples[demonstration].append(accept_ratio)

        normalized_expected_regrets_max_normalization_ma = []
        nevd_max_norm_map_policy = [] # normalizing expected regret with respect to the max for map policy
        nevd_max_norm_mean_policy = [] # normalizing expected regret with respect to the max for mean policy


        _, best_traj_feat_for_map, best_traj_for_map  = \
            trajectory_optimization(env, demonstration_space, trajectory_space, reward_param=map_solution)[0]


        _, best_traj_feat_for_mean, best_traj_for_mean = \
            trajectory_optimization(env, demonstration_space, trajectory_space, reward_param=mean_solution)[0]
        
        map_trajs.append(best_traj_for_map)
        mean_trajs.append(best_traj_for_mean)

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

        true_avar_max_norm_map_policy = ((best_traj_true_reward - map_traj_true_reward) / (2))
        true_avar_bounds_max_norm_map_policy[demonstration + 1].append(true_avar_max_norm_map_policy)
        print(f"True nEVD for {demonstration + 1} demonstration MAP Policy: {true_avar_max_norm_map_policy:.6f}\n")

        true_avar_max_norm_mean_policy = ((best_traj_true_reward - mean_traj_true_reward) / (2))
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

    if experiment_num%20 == 0:
        np.save('results/acceptance_ratio_per_samples_B1_3.3.npy', acceptance_ratio_per_samples)
        np.save('results/mcmc_samples_B1_3.3.npy', mcmc_samples_all_experiments)
        np.save('results/map_trajectories_B1_3.3.npy', np.array(map_trajs))
        np.save('results/mean_trajectories_B1_3.3.npy', np.array(mean_trajs))
        np.save('results/optimal_trajectory_B1_3.3.npy', xi_star)
        np.save('results/avar_all_iterations_max_norm_map_policy_B1_3.3.npy', np.array(avar_all_iterations_max_norm_map_policy))
        np.save('results/true_avar_all_iterations_max_norm_map_policy_B1_3.3.npy', np.array(true_avar_all_iterations_max_norm_map_policy))