import numpy as np
import random
from tqdm import tqdm
import numpy as np
#np.random.seed(42)

def rand_demo(xi):
    n, m = np.shape(xi)
    xi1 = np.copy(xi)
    for idx in range(1, n-1):
        # range of workspace in each axis
        #x = np.random.uniform(0.3, 0.75)
        #y = np.random.uniform(-0.1, 0.5)
        #z = np.random.uniform(0.1, 0.6)
        x = np.random.uniform(-0.1, 0.85)
        y = np.random.uniform(-0.2, 0.7)
        z = np.random.uniform(-0.1, 0.8)
        xi1[idx, :] = np.array([x, y, z])
    return xi1

def p_xi_theta(env, xi, theta, beta=0.1):
    f = env.feature_count(xi, [0, 0, 0])
    R = env.reward(f, theta)
    return beta * R

def human_demo_2(env, xi, theta, n_samples, n_demos=50):
    XI = []
    Fs = []
    Rs = []
    
    best_fs = None
    best_xi = None
    max_reward = np.NINF

    # Add tqdm for progress bar
    for idx in tqdm(range(n_samples), desc="generate traj for optimization..."):
        xi1 = rand_demo(xi)
        f = env.feature_count(xi1, [0, 0, 0])
        
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
    #best_reward = Rs[:n_demos][-1]

    return top_XI, best_xi, top_Fs, best_fs



def mcmc_mean(env, D, n_outer_samples, n_inner_samples, n_burn, xi0, len_theta):
    theta = np.random.rand(len_theta)
    Z_theta = Z_mean(env, xi0, theta, n_inner_samples)
    p_theta = 1.0
    for xi in D:
        expR, _, _, = p_xi_theta(env, xi, theta)
        p_theta *= expR / Z_theta
    theta_samples = []
    for _ in range(n_outer_samples):
        theta_samples.append(theta)
        theta1 = theta + 0.5*(np.random.rand(len_theta)*2-1)
        theta1 = np.clip(theta1, 0, 1)
        Z_theta1 = Z_mean(env, xi0, theta1, n_inner_samples)
        p_theta1 = 1.0
        for xi in D:
            expR1, _, _, = p_xi_theta(env, xi, theta1)
            p_theta1 *= expR1 / Z_theta1
        if p_theta1 / p_theta > np.random.rand():
            theta = np.copy(theta1)
            p_theta = p_theta1
    theta_samples = np.array(theta_samples)
    return theta_samples[-n_burn:,:]

def Z_mean(env, xi, theta, n_samples):
    mean_reward = 0.
    for _ in range(n_samples):
        xi1 = rand_demo(xi)
        expR, _, _, = p_xi_theta(env, xi1, theta)
        mean_reward += expR
    return mean_reward / n_samples

def generate_proposal(prev_theta, step_size, normalize=False):
    # Set the parameters for the 2D normal distribution
    covariance = 0  # covariance between x and y
    cov_matrix = [[step_size**2, covariance, covariance], [covariance, step_size**2, covariance], [covariance, covariance, step_size**2]]
    proposal_r = np.random.multivariate_normal(prev_theta, cov_matrix, 1)[0]
    if normalize:
        proposal_r /= np.linalg.norm(proposal_r)
    return proposal_r

def mcmc_double_2(env, D, n_outer_samples, n_inner_samples, n_burn, len_theta, beta, step_size, normalize=False, adaptive=True):
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
    theta = np.random.randn(len_theta)
    theta_samples = []
    y_prev = (False, None)
    map_ll = -np.inf
    map_theta = None
    n_accepted_sample = 0
    

    stdev = step_size  # initial guess for standard deviation, doesn't matter too much
    accept_cnt = 0  # keep track of how often MCMC accepts

    # For adaptive step sizing
    accept_target = 0.4 # ideally around 40% of the steps accept; if accept count is too high, increase stdev, if too low reduce
    horizon = n_outer_samples // 50 # how often to update stdev based on sliding window avg with window size horizon
    learning_rate = 0.05 # how much to update the stdev
    accept_cnt_list = [] # list of accepts and rejects for sliding window avg


    for i in range(n_outer_samples):
        
        theta1 = generate_proposal(theta, stdev, normalize=normalize)
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
            
            theta_samples.append(theta)
            theta = np.copy(theta1)
            print("Accepted: ", theta1)
            y_prev = (True, y)
            n_accepted_sample += 1

            if p_theta1 > map_ll:
                print("Found MAP solution")
                map_ll = p_theta1
                map_theta = np.copy(theta1)
        else:
            if p_theta1 - p_theta > np.log(epsilon):

                accept_cnt += 1
                if adaptive:
                    accept_cnt_list.append(1)

                print("Randomly accepted: ", theta1)

                #theta_samples.append(theta)
                #n_accepted_sample += 1
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
    print(theta_samples)
    return theta_samples[n_burn:,:], map_theta, n_accepted_sample/n_outer_samples

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

"""
def mcmc_double(env, D, n_outer_samples, n_inner_samples, n_burn, len_theta, beta):

    theta = np.random.rand(len_theta)
    theta_samples = []
    y_prev = (False, None)
    map_ll = -np.inf
    map_theta = None
    

    for _ in range(n_outer_samples):
        #theta_samples.append(theta)
        #theta1 = theta + 0.5*(np.random.rand(len_theta)*2-1)

        #theta1 = theta + 0.5*(np.random.rand(len_theta)*0.5 - 1)
        #theta1 = theta + 0.5*(np.random.rand(len_theta)*0.5-1)

        theta1 = generate_proposal(theta, 0.5)
        #theta1 = np.clip(theta1, 0, 1) # with and without clip ?
        y = inner_sampler(env, D, theta1, n_inner_samples, beta=beta, y_init=y_prev)
        #print("Innnnnnneeeeeeeeer Sampleeeeeeer")
        #print(y)

        #p_theta1 = 1.0
        #p_theta = 1.0
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
        
        #if p_theta1 / p_theta > np.random.rand():
        epsilon = np.random.rand()
        if p_theta1 - p_theta > epsilon:
        #if p_theta1 - p_theta > np.random.rand():
            print("Accept!")
            print(theta1)
            print("Prob: ",p_theta1)
            print("Difference: ", p_theta1 - p_theta)
            print("Epsilon: ", epsilon)
            print(len(theta_samples))
            theta_samples.append(theta)
            theta = np.copy(theta1)
            y_prev = (True, y)
    
            # Update MAP estimate if the new theta1 is greater
            if p_theta1 > map_ll:
                map_ll = p_theta1
                map_theta = np.copy(theta1)
    
    
    theta_samples = np.array(theta_samples)
    return theta_samples[n_burn:,:], map_theta
"""