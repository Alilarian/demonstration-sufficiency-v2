import numpy as np
from algos import *
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.special import logsumexp

def rand_demo(xi):
    n, m = np.shape(xi)
    xi1 = np.copy(xi)
    for idx in range(1, n-1):
        # range of workspace in each axis
        x = np.random.uniform(0.3, 0.75)
        y = np.random.uniform(-0.1, 0.5)
        z = np.random.uniform(0.1, 0.6)
        xi1[idx, :] = np.array([x, y, z])
    return xi1

def p_xi_theta(env, xi, theta, beta=0.1):
    f = env.feature_count(xi)
    R = env.reward(f, theta)
    return beta * R

def human_demo_2(env, xi, theta, n_samples, n_demos=50):
    XI = []
    Fs = []
    Rs = []

    best_fs = None
    best_xi = None
    max_reward = np.NINF
    for idx in range(n_samples):
        xi1 = rand_demo(xi)
        f = env.feature_count(xi1, color=[0, 0, 0])
        if (np.abs(f[0])/np.abs(f[1]) > 50) or (np.abs(f[1])/np.abs(f[0]) > 50):
            print("Bad demo: ", f)
            continue
        else:
            print("Good demo: ", f)
            R = env.reward(f, theta)

            if R > max_reward:
                max_reward = R
                best_fs = f
                best_xi = xi1

            Rs.append(R)
            Fs.append(f)
            XI.append(xi1)
    # Convert lists to numpy arrays
    Rs = np.array(Rs)
    XI = np.array(XI)
    Fs = np.array(Fs)

    # Get indices that would sort Rs in descending order
    sorted_indices = np.argsort(Rs)[::-1]

    # Sort XI, Fs, and Rs based on these indices
    XI = XI[sorted_indices]
    Fs = Fs[sorted_indices]
    Rs = Rs[sorted_indices]

    # Return the top n_demos (default is 10) elements with the highest R
    top_XI = XI[:n_demos]
    top_Fs = Fs[:n_demos]
    top_Rs = Rs[:n_demos]

    #random_traj_feat = np.mean(Fs[n_demos+1:], axis=0)

    return top_XI, best_xi, top_Fs, best_fs, top_Rs

def generate_pairwise_comparisons(trajs_feats, trajs_reward):
    # List to hold the pairwise comparisons
    pairwise_comparisons = []
    # Iterate through all pairs of trajectories
    for i in range(len(trajs_feats)):
        for j in range(len(trajs_feats)):
            if i != j:
                traj_1 = trajs_feats[i]
                traj_2 = trajs_feats[j]

                reward_1 = trajs_reward[i]
                reward_2 = trajs_reward[j]

                # Compare the rewards, add to the result if reward_1 > reward_2
                if reward_1 > reward_2:
                    pairwise_comparisons.append((traj_1, traj_2))
    
    return np.array(pairwise_comparisons)

class PBIRL:

   


    def __init__(self, demonstration, num_features, beta, mcmc_samples, max_reward=0, \
                 mean_reward=0, step_size=1, demo_type='comparison', normalizing='mean'):

        """
        Class for running and storing output of mcmc for Bayesian IRL
        env: the mdp (we ignore the reward)
        demos: list of (s,a) tuples 
        beta: the assumed boltzman rationality of the demonstrator

        """
        #self.demonstrations = demos
        self.demonstration = demonstration
        self.beta = beta
        self.mcmc_samples = mcmc_samples
        self.num_features = num_features
        self.step_size = step_size
        self.accept_ratio = 0
        self.chain_rew = []
        self.map_W = 0
        self.max_reward = max_reward
        self.mean_reward = mean_reward
        self.demo_type = demo_type
        self.normalizing = normalizing


    
    def cal_ll_max(self, w_sample, trajs):
        """
        Aprroximate the denominator of the likelihood with max reward
        """
        log_prior = 0.0  #assume unimformative prior
        log_sum = log_prior

        for traj_index in range(len(trajs)):

            feat_traj = trajs[traj_index]

            reward = feat_traj @ w_sample

            log_sum += self.beta * np.sum(reward) - self.beta * self.max_reward

        return log_sum

    def cal_ll_mean(self, w_sample, trajs):
        """
        Aprroximate the denominator of the likelihood with mean reward
        """
        log_prior = 0.0  #assume unimformative prior
        log_sum = log_prior

        for traj_index in range(len(trajs)):

            feat_traj = trajs[traj_index]

            reward = feat_traj @ w_sample

            log_sum += self.beta * np.sum(reward) - self.beta * self.mean_reward

        return log_sum

    
    
    def calc_ll_pref(self, w_sample, demonstration):

        log_prior = 0.0  #assume unimformative prior
        log_sum = log_prior
        
        for pref_index in range(len(demonstration)): # Each prefrence is a tuple of (i,j) where i is preferred over j. reward(traj_i) > reward(traj_j)
            
            feat_traj1 = demonstration[pref_index][0] # where ind1 is preferred over ind2
            feat_traj2 = demonstration[pref_index][1] # ind1 and ind2 are indexes of corresponding trajectories
            
            #feat_traj1 = trajectories[ind1]
            #feat_traj2 = trajectories[ind2]
            
            #rew1 = calculate_reward_per_step(feat_traj1, w_sample)
            #rew2 = calculate_reward_per_step(feat_traj2, w_sample)
            
            rew1 = feat_traj1 @ w_sample
            rew2 = feat_traj2 @ w_sample
            
            # We can use np.mean instead of sum to see the results
            log_sum += (self.beta * np.sum(rew1)) - np.log(np.exp(self.beta * np.sum(rew1))+ np.exp(self.beta*np.sum(rew2)))

        return log_sum

    
    def generate_proposal(self, prev_theta, stdev,normalize=True):
        # Set the parameters for the 2D normal distribution
        covariance = 0  # covariance between x and y
        cov_matrix = [[stdev**2, covariance, covariance], [covariance, stdev**2, covariance], [covariance, covariance, stdev**2]]
        proposal_r = np.random.multivariate_normal(prev_theta, cov_matrix, 1)[0]
        if normalize:
            proposal_r /= np.linalg.norm(proposal_r)
        return proposal_r

    def initial_solution_bern_cnstr(self):

        # initialize problem solution for MCMC to all zeros, maybe not best initialization but it works in most cases
        #return np.random.randn(self.num_features)
        return np.zeros(self.num_features)

    def run_mcmc(self, adaptive=True):

        '''
            run metropolis hastings MCMC with Gaussian symmetric proposal and uniform prior
            samples: how many reward functions to sample from posterior
            stepsize: standard deviation for proposal distribution
            normalize: if true then it will normalize the rewards (reward weights) to be unit l2 norm, otherwise the rewards will be unbounded
        '''
        
        #mcmc_samples = self.mcmc_samples  # number of MCMC samples

        accept_cnt = 0  #keep track of how often MCMC accepts, ideally around 40% of the steps accept
        #if accept count is too high, increase stdev, if too low reduce

        #self.chain_rew = np.zeros((mcmc_samples, self.num_features))
        #self.chain_rew = []
        cur_rew = self.initial_solution_bern_cnstr()
        cur_W = cur_rew

        cur_sol = cur_W
        # For adaptive step sizing
        accept_target = 0.4 # ideally around 40% of the steps accept; if accept count is too high, increase stdev, if too low reduce
        horizon = self.mcmc_samples // 100 # how often to update stdev based on sliding window avg with window size horizon
        learning_rate = 0.05 # how much to update the stdev
        accept_cnt_list = [] # list of accepts and rejects for sliding window avg
        stdev_list = [] # list of standard deviations for debugging purposes
        accept_prob_list = [] # true cumulative accept probs for debugging purposes
        stdev = self.step_size
        all_lls = [] # all the likelihoods
        
        if self.demo_type == 'comparison':
            cur_ll = self.calc_ll_pref(cur_sol, self.demonstration)  # log likelihood
        
        elif self.demo_type == 'trajectory':
            if self.normalizing == 'max':
                cur_ll = self.cal_ll_max(cur_sol, self.demonstration)
            else:
                cur_ll = self.cal_ll_mean(cur_sol, self.demonstration)


        #keep track of MAP loglikelihood and solution
        map_ll = cur_ll 
        map_W = cur_W
        map_list = []
        Perf_list = []
        for i in range(self.mcmc_samples):
            
            prop_rew = self.generate_proposal(cur_W, stdev,normalize=True)
            
            prop_W = prop_rew
            prop_sol = prop_W
            
            if self.demo_type == 'comparison':
                prop_ll = self.calc_ll_pref(prop_sol, self.demonstration)  # log likelihood
            
            elif self.demo_type == 'trajectory':
                if self.normalizing == 'max':
                    prop_ll = self.cal_ll_max(prop_sol, self.demonstration)
                else:
                    prop_ll = self.cal_ll_mean(prop_sol, self.demonstration)
            

            if prop_ll > cur_ll:
                #print("Yesssssss")
                #self.chain_rew[i,:] = prop_rew
                print('Accepted: ', prop_rew)
                self.chain_rew.append(prop_rew)
                accept_cnt += 1
                cur_W = prop_W
                cur_rew = prop_rew
                cur_ll = prop_ll
                if adaptive:
                    accept_cnt_list.append(1)
                if prop_ll > map_ll:  # maxiumum aposterioi
                    print("HHHHHHHHHooooraaaaaaaaaaaaaaaaaaaHHHHHHHHHHHHHH")
                    map_ll = prop_ll
                    map_W = prop_W
                    map_rew = prop_rew        
            else:
                
                #try:
                #rint("Naive difference:", np.exp(prop_ll - cur_ll))
                if np.random.rand() < np.exp(prop_ll - cur_ll):
                #if np.random.rand() < logsumexp(prop_ll - cur_ll):
                    #print('Noooooo')

                    #self.chain_rew[i,:] = prop_rew
                    self.chain_rew.append(prop_rew)
                    print('Randomly Accepted: ', prop_rew)
                    #accept_cnt += 1
                    cur_W = prop_W
                    cur_rew = prop_rew
                    cur_ll = prop_ll
                    if adaptive:
                        accept_cnt_list.append(1)
                    
                else:
                    # reject               
                    self.chain_rew.append(cur_rew)
                    if adaptive:
                        accept_cnt_list.append(1)
            
            
            if adaptive:
                if len(accept_cnt_list) >= horizon:
                    accept_est = np.sum(accept_cnt_list[-horizon:]) / horizon
                    stdev = max(0.00001, stdev + learning_rate/np.sqrt(i + 1) * (accept_est - accept_target))
                stdev_list.append(stdev)
                accept_prob_list.append(accept_cnt / len(self.chain_rew))
            #if (i+1) % 200 == 0:
                #print("OOOOOOO")
            #    self.step_size -= 0.02

        self.map_rew = map_rew
        self.map_W = map_W
        self.accept_ratio = accept_cnt/self.mcmc_samples

    def get_chain(self, burn_frac=0.1):
        return self.chain_rew[int(burn_frac * len(self.chain_rew)):]
    
    def get_map_solution(self):

        return self.map_W