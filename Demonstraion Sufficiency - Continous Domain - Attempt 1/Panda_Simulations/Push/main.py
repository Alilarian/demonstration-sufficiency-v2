from env_1 import Env1
from algos import *
import numpy as np
import pickle
import argparse
import time

# args parse for run parameters
parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--outer', type=int, default=50)
parser.add_argument('--inner', type=int, default=25)
args = parser.parse_args()

# run parameters
n_runs = args.runs
n_demos = 3
n_outer_samples = args.outer
n_inner_samples = args.inner
n_burn = np.int8(round(n_outer_samples/2,0)) #Number of burned samples from initial sampling


# trajectory parameters
start_position = np.array([0.4, 0.0, 0.5])
goal_position = np.array([0.75, 0.0, 0.3])
xi0 = np.linspace(start_position, goal_position, 4)

# initialize environment
env = Env1(visualize=False)

# performance parameters
error_theta = [0, 0, 0, 0]
regret = [0, 0, 0, 0]
dataset = [[], []]

# for each separate run
for run in range(n_runs):

    # get human demonstrations

    theta = np.random.rand(2)
    print("actual")
    print(theta)
    # D: 3 trajectories, xi_star: traj with largest reward
    D, xi_star, feature_set, f_star = human_demo(env, xi0, theta, 200, n_demos)
    print(D)
    """
    # naive

    print("naive")
    theta_naive = mcmc_naive(env, D, n_outer_samples, n_burn, len(theta))
    theta_naive = np.mean(theta_naive, axis=0)
    regret_naive = get_regret(env, feature_set, theta, f_star, theta_naive)
    print(theta_naive, regret_naive)

    # mean
    print("mean")
    theta_mean = mcmc_mean(env, D, n_outer_samples, n_inner_samples, n_burn, xi0, len(theta))
    theta_mean = np.mean(theta_mean, axis=0)
    regret_mean = get_regret(env, feature_set, theta, f_star, theta_mean)    
    print(theta_mean, regret_mean)

    # max
    print("max")
    theta_max = mcmc_max(env, D, n_outer_samples, n_inner_samples, n_burn, xi0, len(theta))
    theta_max = np.mean(theta_max, axis=0)
    regret_max = get_regret(env, feature_set, theta, f_star, theta_max)
    print(theta_max, regret_max)
    """
    # double
    """ 
    print("double")
    theta_double = mcmc_double(env, D, n_outer_samples, n_inner_samples, n_burn, xi0, len(theta))
    
    regret_per_samples = []
    for theta_i in theta_double:
        regret_per_samples.append(get_regret(env, feature_set, theta, f_star, theta_i))
    
    print(regret_per_samples.sort())
    """
    #theta_double = np.mean(theta_double, axis=0) 
    
    # First calculate the traj* for each theta_i ==> Planning for each theta usingfind_optimal_trajectory(env, theta, traj_length).
                                                # what about starting point and ending point for the traj?
    # Calculate the reward for the optimal traj under each theta => using optimal_features = env.feature_count(optimal_trajectory)
                                                                # and optimal_reward = env.reward(optimal_features, theta)
    # we need to calculate the reward for the traj under theta_map ?????
    

    
    #regret_double = get_regret(env, feature_set, theta, f_star, theta_double)
    


    # metric1 : error in theta
    """ 
    error_theta[0] += np.linalg.norm(theta_naive - theta)
    error_theta[1] += np.linalg.norm(theta_mean - theta)
    error_theta[2] += np.linalg.norm(theta_max - theta)
    error_theta[3] += np.linalg.norm(theta_double - theta)
    learned_thetas = [theta, theta_naive, theta_mean, theta_max, theta_double]
    dataset[0].append(learned_thetas)

    # metric2 : regret
    learned_regrets = [regret_naive, regret_mean, regret_max, regret_double]
    dataset[1].append(learned_regrets)

    # report the progress
    e1 = round(error_theta[0], 1)
    e2 = round(error_theta[1], 1)
    e3 = round(error_theta[2], 1)
    e4 = round(error_theta[3], 1)
    savestring = "data/error_push" + str(n_inner_samples) +"_"+ str(n_outer_samples)+".pkl"
    pickle.dump(dataset, open(savestring, "wb"))
"""