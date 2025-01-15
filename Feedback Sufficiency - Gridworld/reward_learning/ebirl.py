from agent.q_learning_agent import ValueIteration
import numpy as np
import copy
from utils.common_helper import compute_reward_for_trajectory


class EBIRL:
    def __init__(self, env, demos, beta, epsilon=0.0001):

        """
        Class for running and storing output of mcmc for Bayesian IRL
        env: the mdp (we ignore the reward)
        demos: list of (s,a) tuples 
        beta: the assumed boltzman rationality of the demonstrator

        """
        self.env = copy.deepcopy(env)
        self.demonstrations = demos
        self.epsilon = epsilon
        self.beta = beta
        self.value_iters = {} # key: environment, value: value iteration result
        self.num_mcmc_dims = len(self.env.feature_weights)

    def generate_proposal(self, old_sol, stdev, normalize):
        """
        Symetric Gaussian proposal
        """
        proposal_r = old_sol + stdev * np.random.randn(len(old_sol))
        if normalize:
            proposal_r /= np.linalg.norm(proposal_r)
        return proposal_r

    def initial_solution(self):
        # initialize problem solution for MCMC to all zeros, maybe not best initialization but it works in most cases
        #return np.zeros(self.num_mcmc_dims)
        
        init = np.random.randn(self.num_mcmc_dims)
        return init / np.linalg.norm(init)

    def calc_ll(self, hyp_reward):

        self.env.set_feature_weights(hyp_reward)
        # Initialize the log_prior as 0, assuming an uninformative prior
        log_prior = 0.0
        log_sum = log_prior  # Start the log sum with the log prior value

        for estop in self.demonstrations:

            # comput the reward until step ti (stopping point)

            # compute reward for all time step in that trajectory (denominator)
            # Sub-trajectory ξ_0:t (use cumulative rewards up to point t)

            trajectory ,t = estop
            traj_len = len(trajectory)
            reward_up_to_t = sum(self.env.compute_reward(s) for s, _ in trajectory[:t+1])
            
            stop_prob_numerator = self.beta * reward_up_to_t

            #print("Nominator: ", stop_prob_numerator)
            
            
            # Compute denominator (normalization factor for the entire trajectory)
            stop_prob_denominator = sum(np.exp(self.beta * sum(self.env.compute_reward(s) for s, _ in trajectory[:k+1])) for k in range(traj_len))
            
            log_sum += stop_prob_numerator - np.log(stop_prob_denominator)
            #print("denominator: ", np.log(stop_prob_denominator))

        return log_sum
        
    def run_mcmc(self, samples, stepsize, normalize=True, adaptive=False):
        '''
            run metropolis hastings MCMC with Gaussian symmetric proposal and uniform prior
            samples: how many reward functions to sample from posterior
            stepsize: standard deviation for proposal distribution
            normalize: if true then it will normalize the rewards (reward weights) to be unit l2 norm, otherwise the rewards will be unbounded
        '''
        num_samples = samples  # number of MCMC samples
        stdev = stepsize  # initial guess for standard deviation, doesn't matter too much
        accept_cnt = 0  # keep track of how often MCMC accepts

        # For adaptive step sizing
        accept_target = 0.4 # ideally around 40% of the steps accept; if accept count is too high, increase stdev, if too low reduce
        horizon = num_samples // 100 # how often to update stdev based on sliding window avg with window size horizon
        learning_rate = 0.05 # how much to update the stdev
        accept_cnt_list = [] # list of accepts and rejects for sliding window avg
        stdev_list = [] # list of standard deviations for debugging purposes
        accept_prob_list = [] # true cumulative accept probs for debugging purposes
        all_lls = [] # all the likelihoods

        self.chain = np.zeros((num_samples, self.num_mcmc_dims)) #store rewards found via BIRL here, preallocate for speed
        self.likelihoods = [0 for _ in range(num_samples)]
        cur_sol = self.initial_solution() #initial guess for MCMC
        cur_ll = self.calc_ll(cur_sol)  # log likelihood
        #keep track of MAP loglikelihood and solution
        map_ll = cur_ll
        map_sol = cur_sol

        for i in range(num_samples):
            # sample from proposal distribution
            prop_sol = self.generate_proposal(cur_sol, stdev, normalize)
            # calculate likelihood ratio test
            prop_ll = self.calc_ll(prop_sol)
            all_lls.append(prop_ll)
            if prop_ll > cur_ll:
                # accept
                self.chain[i, :] = prop_sol
                self.likelihoods[i] = prop_ll
                accept_cnt += 1
                if adaptive:
                    accept_cnt_list.append(1)
                cur_sol = prop_sol
                cur_ll = prop_ll
                if prop_ll > map_ll:  # maxiumum aposterioi
                    map_ll = prop_ll
                    map_sol = prop_sol
                    print("MAP INSIDE the MCMC: ", map_sol)
            else:
                # accept with prob exp(prop_ll - cur_ll)
                if np.random.rand() < np.exp(prop_ll - cur_ll):
                    self.chain[i, :] = prop_sol
                    self.likelihoods[i] = prop_ll
                    accept_cnt += 1
                    if adaptive:
                        accept_cnt_list.append(1)
                    cur_sol = prop_sol
                    cur_ll = prop_ll
                else:
                    # reject
                    self.chain[i, :] = cur_sol
                    self.likelihoods[i] = cur_ll
                    if adaptive:
                        accept_cnt_list.append(0)
            # Check for step size adaptation
            if adaptive:
                if len(accept_cnt_list) >= horizon:
                    accept_est = np.sum(accept_cnt_list[-horizon:]) / horizon
                    stdev = max(0.00001, stdev + learning_rate/np.sqrt(i + 1) * (accept_est - accept_target))
                stdev_list.append(stdev)
                accept_prob_list.append(accept_cnt / len(self.chain))
        self.accept_rate = accept_cnt / num_samples
        self.map_sol = map_sol
        
    def get_map_solution(self):
        return self.map_sol

    def get_mean_solution(self, burn_frac=0.1, skip_rate=1):
        ''' get mean solution after removeing burn_frac fraction of the initial samples and only return every skip_rate
            sample. Skiping reduces the size of the posterior and can reduce autocorrelation. Burning the first X% samples is
            often good since the starting solution for mcmc may not be good and it can take a while to reach a good mixing point
        '''

        burn_indx = int(len(self.chain) * burn_frac)
        mean_r = np.mean(self.chain[burn_indx::skip_rate], axis=0)
        
        return mean_r