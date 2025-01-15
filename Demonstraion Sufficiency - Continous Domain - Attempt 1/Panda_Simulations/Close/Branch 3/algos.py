import numpy as np
import random


def p_xi_theta(env, xi, theta, beta=20.0):
    f = env.feature_count(xi)
    R = env.reward(f, theta)
    #return np.exp(beta * R), f, R
    return beta * R

def rand_demo(xi):
    n, m = np.shape(xi)
    xi1 = np.copy(xi)
    for idx in range(1, n):
        # range of workspace in each axis
        x = np.random.uniform(0.3, 0.75)
        y = np.random.uniform(-0.5, 0.5)
        z = np.random.uniform(0.1, 0.6)
        xi1[idx, :] = np.array([x, y, z])
    return xi1

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

    return top_XI, best_xi, top_Fs, best_fs

def generate_proposal(prev_theta, step_size, normalize=False):

    # Set the parameters for the 2D normal distribution


    covariance = 0  # covariance between x and y

    cov_matrix = [[step_size**2, covariance], [covariance, step_size**2]]

    proposal_r = np.random.multivariate_normal(prev_theta, cov_matrix, 1)[0]

    if normalize:
        proposal_r /= np.linalg.norm(proposal_r)


    return proposal_r

def mcmc_double_2(env, D, n_outer_samples, n_inner_samples, n_burn, len_theta, beta, step_size):
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
    #theta = np.random.rand(len_theta)
    theta = np.zeros(len_theta)
    theta_samples = []
    y_prev = (False, None)
    map_ll = -np.inf
    map_theta = None
    accep_ratio = 0
    

    for _ in range(n_outer_samples):

        theta1 = generate_proposal(theta, step_size, normalize=False)
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
        
        #if p_theta1 / p_theta > np.random.rand():
        epsilon = np.random.rand()
        if p_theta1 > p_theta:
            
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
            print(p_theta1 - p_theta)
            #print(epsilon)
            print(np.log(epsilon))
            if p_theta1 - p_theta > np.log(epsilon):
            
                print("Randomly Accepted!")
                print("New param: ", theta1)
                theta_samples.append(theta)
                accep_ratio += 1
                theta = np.copy(theta1)
                y_prev = (True, y)
        
            else:
                theta = theta

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