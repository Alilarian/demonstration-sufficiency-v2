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

#trajectory parameters
start_position = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
goal_position = np.array([0.288, 0.128, 0.159, -1.752, -2.256, 2.736, -0.096])
xi0 = np.linspace(start_position, goal_position, 3)

# initialize environment
env = Env1(visualize=False)

# performance parameters
error_theta = [0, 0, 0, 0]
regret = [0, 0, 0, 0]
time_mat = [0, 0, 0, 0]
dataset = [[], []]

# for each separate run
for run in range(n_runs):

    # get human demonstrations
    theta = np.random.rand(2)
    print("actual")
    print(theta)
    D, xi_star, feature_set, f_star = human_demo(env, xi0, theta, 200, n_demos)

    # naive
    print("naive")
    #temp1 = time.time()
    theta_naive = mcmc_naive(env, D, n_outer_samples*n_inner_samples, n_burn, len(theta))
    #temp2 = time.time() - temp1
    time_mat[0] += temp2
    theta_naive = np.mean(theta_naive, axis=0)
    regret_naive = get_regret(env, feature_set, theta, f_star, theta_naive)
    print(theta_naive, regret_naive)#,temp2)

    # mean
    print("mean")
    temp1 = time.time()
    theta_mean = mcmc_mean(env, D, n_outer_samples, n_inner_samples, n_burn, xi0, len(theta))
    temp2 = time.time() - temp1
    time_mat[1] += temp2
    theta_mean = np.mean(theta_mean, axis=0)
    regret_mean = get_regret(env, feature_set, theta, f_star, theta_mean)    
    print(theta_mean, regret_mean)#,temp2)

    # max
    print("max")
    #temp1 = time.time()
    theta_max = mcmc_max(env, D, n_outer_samples, n_inner_samples, n_burn, xi0, len(theta))
    #temp2 = time.time() - temp1
    #time_mat[2] += temp2
    theta_max = np.mean(theta_max, axis=0)
    regret_max = get_regret(env, feature_set, theta, f_star, theta_max)
    print(theta_max, regret_max)#,temp2)

    # double
    print("double")
    #temp1 = time.time()
    theta_double = mcmc_double(env, D, n_outer_samples, n_inner_samples, n_burn, xi0, len(theta))
    #temp2 = time.time() - temp1
    #time_mat[3] += temp2
    theta_double = np.mean(theta_double, axis=0)
    regret_double = get_regret(env, feature_set, theta, f_star, theta_double)    
    print(theta_double, regret_double)#,temp2)

    # metric1 : error in theta
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
    print("[*] iter", run, "error", [e1, e2, e3, e4])
    savestring = "data/error_pour" + str(n_inner_samples) +"_"+ str(n_outer_samples)+".pkl"
    pickle.dump(dataset, open(savestring, "wb"))
#savestring2 = "data/rebuttal/error_s3_time1" + str(n_inner_samples) +"_"+ str(n_outer_samples)+".pkl"
#pickle.dump(time_mat, open(savestring2, "wb"))
