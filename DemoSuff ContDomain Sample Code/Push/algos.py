import numpy as np
import random

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
    #return np.exp(beta * R)
    return beta * R

def human_demo_2(env, xi, theta, n_samples, n_demos):
    XI = []
    Fs = []
    Rs = []
    
    best_fs = None
    best_xi = None
    max_reward = np.NINF
    for idx in range(n_samples):
        xi1 = rand_demo(xi)
        
        f = env.feature_count(xi1)

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

    Rs = np.array(Rs)
    XI = np.array(XI)
    Fs = np.array(Fs)

    # Remove best_xi and best_fs from XI and Fs respectively
    if best_xi is not None and best_fs is not None:
        XI = np.array([x for x in XI if not np.array_equal(x, best_xi)])
        Fs = np.array([f for f in Fs if not np.array_equal(f, best_fs)])

    return XI, best_xi, Fs, best_fs

def generate_proposal(prev_theta, step_size, normalize=False):

    # Set the parameters for the 2D normal distribution
    covariance = 0  # covariance between x and y

    cov_matrix = [[step_size**2, covariance], [covariance, step_size**2]]

    proposal_r = np.random.multivariate_normal(prev_theta, cov_matrix, 1)[0]

    if normalize:
        proposal_r /= np.linalg.norm(proposal_r)


    return proposal_r

def mcmc_double_2(env, D, n_outer_samples, n_inner_samples, n_burn, len_theta, beta, step_size, adaptive=True):
    """
    Perform MCMC sampling to estimate the parameter theta.

    Parameters:
    - env: The environment for the trajectories.
    - D: List of observed trajectories.
    - n_outer_samples: Number of outer loop MCMC samples.
    - n_inner_samples: Number of inner loop MCMC samples.
    - n_burn: Number of initial samples to discard (burn-in period).
    - len_theta: Length of the parameter vector theta.

    Returns:
    - theta_samples: Array of sampled theta values after burn-in.
    - map_theta: Theta value with the highest likelihood (MAP estimate).
    """
    theta = np.zeros(len_theta)
    
    theta_samples = []
    y_prev = (False, None)
    map_ll = -np.inf
    map_theta = None
    accep_ratio = 0
    
    stdev = step_size  # initial guess for standard deviation, doesn't matter too much
    accept_cnt = 0  # keep track of how often MCMC accepts
    # For adaptive step sizing
    accept_target = 0.4 # ideally around 40% of the steps accept; if accept count is too high, increase stdev, if too low reduce
    horizon = n_outer_samples // 10 # how often to update stdev based on sliding window avg with window size horizon
    learning_rate = 0.05 # how much to update the stdev
    accept_cnt_list = [] # list of accepts and rejects for sliding window avg

    for i in range(n_outer_samples):
        
        theta1 = generate_proposal(theta, step_size, normalize=False)
        #if (np.abs(theta1[0])/np.abs(theta1[1]) > 12) or np.abs(theta1[1])/np.abs(theta1[0]) > 12)
        
        #theta1 = np.clip(theta1, 0, 1) # with and without clip ?
        y = inner_sampler(env, D, theta1, n_inner_samples, beta=beta, y_init=y_prev)
        p_theta1 = 0
        p_theta = 0
        for xi in D:
            expRx1 = p_xi_theta(env, xi, theta1, beta=beta)
            expRy1 = p_xi_theta(env, y, theta1, beta=beta)
            
            p_theta1 += (expRx1 - expRy1)
        
        for xi in D:
            expRx  = p_xi_theta(env, xi, theta, beta=beta)
            expRy = p_xi_theta(env, y, theta, beta=beta)
            p_theta += (expRx - expRy)
        
        epsilon = np.random.rand()
        
        if p_theta1 > p_theta:
            
            accept_cnt += 1
            if adaptive:
                accept_cnt_list.append(1)
            #print("Accept!")
            #print("New likelihood: ", theta1)
            theta_samples.append(theta)
            theta = np.copy(theta1)
            print("Accepted")
            print(theta)
            y_prev = (True, y)
            accep_ratio += 1

            if p_theta1 > map_ll:
                map_ll = p_theta1
                map_theta = np.copy(theta1)
        else:
            if p_theta1 - p_theta > np.log(epsilon):

                accept_cnt += 1
                if adaptive:
                    accept_cnt_list.append(1)
            
                print("Randomly Accepted!")
                print("New param: ", theta1)
                theta_samples.append(theta)
                accep_ratio += 1
                theta = np.copy(theta1)
                y_prev = (True, y)
        
            else:
                theta = theta
        
        if adaptive:
            if len(accept_cnt_list) >= horizon:
                accept_est = np.sum(accept_cnt_list[-horizon:]) / horizon
                stdev = max(0.00001, stdev + learning_rate/np.sqrt(i + 1) * (accept_est - accept_target))
         
    theta_samples = np.array(theta_samples)
    
    n_burn = int(np.ceil(n_burn * len(theta_samples)))
    
    return theta_samples[n_burn:,:], map_theta, accep_ratio/n_outer_samples


def inner_sampler(env, D, theta, n_samples, beta, y_init=(False, None)):
    if y_init[0]:
        y = y_init[1]
    else:
        y = random.choice(D)

    y_score = p_xi_theta(env, y, theta)        
    
    for _ in range(n_samples):
        y1 = rand_demo(y)
        y1_score = p_xi_theta(env, y1, theta, beta=beta)
        if y1_score -  y_score > np.random.rand():
            y = np.copy(y1)
            y_score = y1_score
    return y