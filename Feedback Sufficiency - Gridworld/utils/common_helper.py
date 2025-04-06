import sys
import os
import time
import yaml
import numpy as np
import random

import copy
import math
from scipy.stats import norm

# Get current and parent directory to handle import paths
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from agent.q_learning_agent import ValueIteration, PolicyEvaluation

def calculate_percentage_optimal_actions(policy, env, epsilon=0.0001):
    """
    Calculate the percentage of actions in the given policy that are optimal under the environment's Q-values.

    Args:
        policy (list): List of actions for each state.
        env: The environment object.
        epsilon (float): Tolerance for determining optimal actions.

    Returns:
        float: Percentage of optimal actions in the policy.
    """
    # Compute Q-values using value iteration
    q_values = ValueIteration(env).get_q_values()
    
    # Count how many actions in the policy are optimal under the environment
    optimal_actions_count = sum(
        1 for state, action in enumerate(policy) if action in _arg_max_set(q_values[state], epsilon)
    )
    
    return optimal_actions_count / env.num_states

def _arg_max_set(values, epsilon=0.0001):
    """
    Returns the indices corresponding to the maximum element(s) in a set of values, within a tolerance.

    Args:
        values (list or np.array): List of values to evaluate.
        epsilon (float): Tolerance for determining equality of maximum values.

    Returns:
        list: Indices of the maximum value(s).
    """
    max_val = max(values)
    return [i for i, v in enumerate(values) if abs(max_val - v) < epsilon]

def calculate_expected_value_difference(eval_policy, env, epsilon=0.0001, normalize_with_random_policy=False):
    """
    Calculates the difference in expected returns between an optimal policy for an MDP and the eval_policy.

    Args:
        eval_policy (list): The policy to evaluate.
        env: The environment object.
        storage (dict): A storage dictionary (not used in this version, but passed for consistency).
        epsilon (float): Convergence threshold for value iteration and policy evaluation.
        normalize_with_random_policy (bool): Whether to normalize using a random policy.

    Returns:
        float: The difference in expected returns between the optimal policy and eval_policy.
    """
    
    # Run value iteration to get the optimal state values
    V_opt = ValueIteration(env).run_value_iteration(epsilon=epsilon)
    
    # Perform policy evaluation for the provided eval_policy
    V_eval = PolicyEvaluation(env, policy=eval_policy).run_policy_evaluation(epsilon=epsilon)
    
    # Optional: Normalize using a random policy if the flag is set
    if normalize_with_random_policy:
        V_rand = PolicyEvaluation(env, uniform_random=True).run_policy_evaluation(epsilon=epsilon)
        #if np.mean(V_opt) - np.mean(V_eval) == 0:
        #    return 0.0

        return (np.mean(V_opt) - np.mean(V_eval)) / (np.mean(V_opt) - np.mean(V_rand))
        #return (np.mean(V_opt) - np.mean(V_eval)) / (np.mean(V_opt))

    # Return the unnormalized difference in expected returns between optimal and eval_policy
    return np.mean(V_opt) - np.mean(V_eval)

def calculate_policy_accuracy(opt_pi, eval_pi):
    assert len(opt_pi) == len(eval_pi)
    matches = 0
    for i in range(len(opt_pi)):
        matches += opt_pi[i] == eval_pi[i]
    return matches / len(opt_pi)
'''
def compute_policy_loss_avar_bound(mcmc_samples, env, map_policy, random_normalization, alpha, delta):

    policy_losses = []

    # Step 1: Calculate policy loss for each MCMC sample
    for sample in mcmc_samples:
        learned_env = copy.deepcopy(env)  # Create a copy of the environment
        learned_env.set_feature_weights(sample)   # Set the reward function to the current sample
        
        # Calculate the policy loss (Expected Value Difference)
        policy_loss = calculate_expected_value_difference(
            map_policy, learned_env, normalize_with_random_policy=random_normalization
        )
        policy_losses.append(policy_loss)

    # Step 2: Sort the policy losses
    policy_losses.sort()

    # Step 3: Compute the VaR (Value at Risk) bound
    N_burned = len(mcmc_samples)
    k = math.ceil(N_burned * alpha + norm.ppf(1 - delta) * np.sqrt(N_burned * alpha * (1 - alpha)) - 0.5)
    k = min(k, N_burned - 1)  # Ensure k doesn't exceed the number of samples

    # Return the computed a-VaR bound
    return policy_losses[k]
'''
def compute_policy_loss_avar_bounds(mcmc_samples, env, map_policy, random_normalization, alphas, delta):
    """
    Computes the counterfactual policy losses and calculates the a-VaR (Value at Risk) bound for multiple alpha values.

    Args:
        mcmc_samples (list): List of MCMC sampled rewards from the BIRL process.
        env: The environment object.
        map_policy: The MAP (Maximum a Posteriori) policy from BIRL.
        random_normalization (bool): Whether to normalize using a random policy.
        alphas (list of float): List of confidence level parameters.
        delta (float): Risk level parameter.

    Returns:
        dict: A dictionary mapping each alpha to its computed a-VaR bound.
    """
    policy_losses = []

    # Step 1: Calculate policy loss for each MCMC sample
    for sample in mcmc_samples:
        learned_env = copy.deepcopy(env)  # Create a copy of the environment
        learned_env.set_feature_weights(sample)   # Set the reward function to the current sample
        
        # Calculate the policy loss (Expected Value Difference)
        policy_loss = calculate_expected_value_difference(
            map_policy, learned_env, normalize_with_random_policy=random_normalization
        )
        policy_losses.append(policy_loss)

    # Step 2: Sort the policy losses
    policy_losses.sort()

    # Step 3: Compute the a-VaR bound for each alpha
    N_burned = len(mcmc_samples)
    avar_bounds = {}

    for alpha in alphas:
        k = math.ceil(N_burned * alpha + norm.ppf(1 - delta) * np.sqrt(N_burned * alpha * (1 - alpha)) - 0.5)
        k = min(k, N_burned - 1)  # Ensure k doesn't exceed the number of samples
        avar_bounds[alpha] = policy_losses[k]

    return avar_bounds

def compute_reward_for_trajectory(env, trajectory, discount_factor=None):
    """
    Computes the cumulative reward for a given trajectory in the environment. If a discount factor 
    is provided, the function calculates the discounted cumulative reward, where rewards received 
    later in the trajectory are given less weight.

    :param env: The environment (MDP) which provides the reward function. This should have a 
                `compute_reward(state)` method.
    :param trajectory: List of tuples (state, action), where `state` is the current state and 
                       `action` is the action taken in that state. The action is ignored in reward 
                       computation but kept for compatibility with the trajectory format.
    :param discount_factor: (Optional) A float representing the discount factor (gamma) for 
                            future rewards. It should be between 0 and 1. If None, no discounting 
                            is applied, and rewards are summed without any decay.
    :return: The cumulative reward for the trajectory, either discounted or non-discounted.
             If discount_factor is provided, it applies a discount based on the time step of 
             the trajectory.
    """
    cumulative_reward = 0
    discount = 1 if discount_factor is None else discount_factor
    
    for t, (state, action) in enumerate(trajectory):
        if state is None:  # Terminal state reached
            break
        
        # Compute the reward for the current state
        reward = env.compute_reward(state)
        
        # If a discount factor is provided, apply it to the reward
        if discount_factor:
            cumulative_reward += reward * (discount_factor ** t)
        else:
            cumulative_reward += reward

    return cumulative_reward

def logsumexp(x):
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))

import numpy as np
from scipy.special import logsumexp

def compute_infogain(env, demos, mcmc_samples_1, mcmc_samples_2, beta, log_prob_func):
    """
    Compute information gain between posterior and prior MCMC samples for demonstrations.
    
    Args:
        env: The GridWorld environment.
        demos: List of demonstrations (state-action pairs or preferences).
        mcmc_samples_1: MCMC samples from prior \( \Theta_{n-1} \).
        mcmc_samples_2: MCMC samples from posterior \( \Theta_n \).
        beta: Rationality parameter.
        log_prob_func: Function to compute log probability (either log_prob_demo or log_prob_comparison).
    
    Returns:
        float: Information gain value.
    """
    M1 = len(mcmc_samples_1)  # Number of prior samples
    M2 = len(mcmc_samples_2)  # Number of posterior samples

    # Handle initial condition (n=1)
    if len(demos) == 1:
        # Compute denominator using logsumexp for numerical stability
        posterior_denominator = logsumexp([log_prob_func(env, demos, theta, beta) for theta in mcmc_samples_2])

        print("posterior_denominator: ", posterior_denominator)

        second_term = 0
        for theta_posterior in mcmc_samples_2:
            log_p_demos_posterior = log_prob_func(env, demos, theta_posterior, beta)
            second_term += log_p_demos_posterior - posterior_denominator + np.log(M2)
        second_term /= M2

        return second_term

    # Compute log probabilities for prior and posterior samples
    prior_log_probs = np.array([log_prob_func(env, demos[:-1], theta, beta) for theta in mcmc_samples_1])
    posterior_log_probs = np.array([log_prob_func(env, demos, theta, beta) for theta in mcmc_samples_2])

    # Compute denominators using logsumexp
    prior_denominator = logsumexp(prior_log_probs)
    posterior_denominator = logsumexp(posterior_log_probs)

    # Compute first term: Expectation over prior samples
    first_term = np.mean(prior_denominator - prior_log_probs - np.log(M1))

    # Compute second term: Expectation over posterior samples
    second_term = np.mean(posterior_log_probs - posterior_denominator + np.log(M2))

    # Compute total information gain
    info_gain = first_term + second_term

    return info_gain


def compute_infogain_2(env, demos, mcmc_samples_1, mcmc_samples_2, beta, log_prob_func):
    """
    Compute information gain between posterior and prior MCMC samples for demonstrations.
    
    Args:
        env: The GridWorld environment.
        demos: List of demonstrations (state-action pairs or preferences).
        mcmc_samples_1: MCMC samples from prior \( \Theta_{n-1} \).
        mcmc_samples_2: MCMC samples from posterior \( \Theta_n \).
        beta: Rationality parameter.
        log_prob_func: Function to compute log probability (either log_prob_demo or log_prob_comparison).
    
    Returns:
        float: Information gain value.
    """
    M1 = len(mcmc_samples_1)  # Number of prior samples
    M2 = len(mcmc_samples_2)  # Number of posterior samples

    # Handle initial condition (n=1)
    if len(demos) == 1:
        posterior_log_probs = np.exp(np.array([log_prob_func(env, demos, theta, beta) for theta in mcmc_samples_2]))
        
        ## Apply log on each component of posterior_log_probs
        ## Sum them and take average
        second_term = np.mean(np.log(posterior_log_probs)) + np.log(M2) - np.log(np.sum(posterior_log_probs))
        
        #posterior_denominator = logsumexp(posterior_log_probs)

        #second_term = np.mean(posterior_log_probs - posterior_denominator + np.log(M2))
        return second_term

    # Compute log probabilities for prior and posterior samples
    prior_log_probs = np.exp(np.array([log_prob_func(env, demos[:-1], theta, beta) for theta in mcmc_samples_1]))
    posterior_log_probs = np.exp(np.array([log_prob_func(env, demos, theta, beta) for theta in mcmc_samples_2]))

    second_term = np.mean(np.log(posterior_log_probs)) + np.log(M2) - np.log(np.sum(posterior_log_probs))

    first_term = np.log(np.sum(prior_log_probs)) - np.log(M1) - np.mean(np.log(prior_log_probs))

    # Compute total information gain
    info_gain = first_term + second_term

    return info_gain

def compute_infogain_3(env, demos, mcmc_samples_1, mcmc_samples_2, beta, log_prob_func):
    """
    Compute information gain between posterior and prior MCMC samples for demonstrations.
    
    Args:
        env: The GridWorld environment.
        demos: List of demonstrations (state-action pairs or preferences).
        mcmc_samples_1: MCMC samples from prior \( \Theta_{n-1} \).
        mcmc_samples_2: MCMC samples from posterior \( \Theta_n \).
        beta: Rationality parameter.
        log_prob_func: Function to compute log probability (either log_prob_demo or log_prob_comparison).
    
    Returns:
        float: Information gain value.
    """
    M1 = len(mcmc_samples_1)  # Number of prior samples
    M2 = len(mcmc_samples_2)  # Number of posterior samples

    # Handle initial condition (n=1)
    if len(demos) == 1:
        posterior_log_probs = np.exp(np.array([log_prob_func(env, demos, theta, beta) for theta in mcmc_samples_2]))
        
        ## Apply log on each component of posterior_log_probs
        ## Sum them and take average
        second_term = np.mean(np.log(posterior_log_probs)) + np.log(M2) - np.log(np.sum(posterior_log_probs))
        
        #posterior_denominator = logsumexp(posterior_log_probs)

        #second_term = np.mean(posterior_log_probs - posterior_denominator + np.log(M2))
        return second_term

    # Compute log probabilities for prior and posterior samples
    prior_log_probs = np.exp(np.array([log_prob_func(env, demos[:-1], theta, beta) for theta in mcmc_samples_1]))
    posterior_log_probs = np.exp(np.array([log_prob_func(env, demos, theta, beta) for theta in mcmc_samples_2]))

    second_term = np.mean(np.log(posterior_log_probs)) - np.log(np.sum(posterior_log_probs))

    first_term = np.log(np.sum(prior_log_probs)) - np.mean(np.log(prior_log_probs))

    # Compute total information gain
    info_gain = first_term + second_term

    return info_gain

def compute_infogain_4(env, demos, mcmc_samples_1, mcmc_samples_2, beta, log_prob_func):
    """
    Compute information gain between posterior and prior MCMC samples for demonstrations.
    
    Args:
        env: The GridWorld environment.
        demos: List of demonstrations (state-action pairs or preferences).
        mcmc_samples_1: MCMC samples from prior \( \Theta_{n-1} \).
        mcmc_samples_2: MCMC samples from posterior \( \Theta_n \).
        beta: Rationality parameter.
        log_prob_func: Function to compute log probability (either log_prob_demo or log_prob_comparison).
    
    Returns:
        float: Information gain value.
    """
    M1 = len(mcmc_samples_1)  # Number of prior samples
    M2 = len(mcmc_samples_2)  # Number of posterior samples

    # Handle initial condition (n=1)
    if len(demos) == 1:
        posterior_log_probs = np.array([log_prob_func(env, demos, theta, beta) for theta in mcmc_samples_2])
        
        ## Apply log on each component of posterior_log_probs
        ## Sum them and take average
        second_term = np.mean(posterior_log_probs) + np.log(M2) - logsumexp(posterior_log_probs)
        #posterior_denominator = logsumexp(posterior_log_probs)

        #second_term = np.mean(posterior_log_probs - posterior_denominator + np.log(M2))
        return second_term

    # Compute log probabilities for prior and posterior samples
    prior_log_probs = np.array([log_prob_func(env, demos[:-1], theta, beta) for theta in mcmc_samples_1])
    posterior_log_probs = np.array([log_prob_func(env, demos, theta, beta) for theta in mcmc_samples_2])

    second_term = np.mean(posterior_log_probs) + np.log(M2) - logsumexp(posterior_log_probs)

    first_term = logsumexp(prior_log_probs) - np.log(M1) - np.mean(prior_log_probs)

    # Compute total information gain
    info_gain = first_term + second_term

    return info_gain


def compute_infogain_5(env, demos, mcmc_samples_1, mcmc_samples_2, beta, log_prob_func):
    """
    Compute information gain between posterior and prior MCMC samples for demonstrations.
    
    Args:
        env: The GridWorld environment.
        demos: List of demonstrations (state-action pairs or preferences).
        mcmc_samples_1: MCMC samples from prior \( \Theta_{n-1} \).
        mcmc_samples_2: MCMC samples from posterior \( \Theta_n \).
        beta: Rationality parameter.
        log_prob_func: Function to compute log probability (either log_prob_demo or log_prob_comparison).
    
    Returns:
        float: Information gain value.
    """
    M1 = len(mcmc_samples_1)  # Number of prior samples
    M2 = len(mcmc_samples_2)  # Number of posterior samples

    # Handle initial condition (n=1)
    if len(demos) == 1:
        posterior_log_probs = np.exp(np.array([log_prob_func(env, demos, theta, beta) for theta in mcmc_samples_2]))
        
        ## Apply log on each component of posterior_log_probs
        ## Sum them and take average
        second_term = np.mean(posterior_log_probs) + np.log(M2) - logsumexp(posterior_log_probs)
        #posterior_denominator = logsumexp(posterior_log_probs)

        #second_term = np.mean(posterior_log_probs - posterior_denominator + np.log(M2))
        return second_term

    # Compute log probabilities for prior and posterior samples
    prior_log_probs = np.exp(np.array([log_prob_func(env, demos[:-1], theta, beta) for theta in mcmc_samples_1]))
    posterior_log_probs = np.exp(np.array([log_prob_func(env, demos, theta, beta) for theta in mcmc_samples_2]))

    second_term = np.mean(posterior_log_probs) + np.log(M2) - logsumexp(posterior_log_probs)

    first_term = logsumexp(prior_log_probs) - np.log(M1) - np.mean(prior_log_probs)

    # Compute total information gain
    info_gain = first_term + second_term

    return info_gain

def compute_infogain_6(env, demos, mcmc_samples_1, mcmc_samples_2, beta, log_prob_func):
    """
    Compute information gain between posterior and prior MCMC samples for demonstrations.
    
    Args:
        env: The GridWorld environment.
        demos: List of demonstrations (state-action pairs or preferences).
        mcmc_samples_1: MCMC samples from prior \( \Theta_{n-1} \).
        mcmc_samples_2: MCMC samples from posterior \( \Theta_n \).
        beta: Rationality parameter.
        log_prob_func: Function to compute log probability (either log_prob_demo or log_prob_comparison).
    
    Returns:
        float: Information gain value.
    """
    M1 = len(mcmc_samples_1)  # Number of prior samples
    M2 = len(mcmc_samples_2)  # Number of posterior samples

    # Handle initial condition (n=1)
    if len(demos) == 1:
        posterior_log_probs = np.array([log_prob_func(env, demos, theta, beta) for theta in mcmc_samples_2])
        
        ## Apply log on each component of posterior_log_probs
        ## Sum them and take average
        second_term = M2* (np.mean(posterior_log_probs) + np.log(M2) - logsumexp(posterior_log_probs))
        #posterior_denominator = logsumexp(posterior_log_probs)

        #second_term = np.mean(posterior_log_probs - posterior_denominator + np.log(M2))
        return second_term

    # Compute log probabilities for prior and posterior samples
    prior_log_probs = np.array([log_prob_func(env, demos[:-1], theta, beta) for theta in mcmc_samples_1])
    posterior_log_probs = np.array([log_prob_func(env, demos, theta, beta) for theta in mcmc_samples_2])

    second_term = np.mean(posterior_log_probs) + np.log(M2) - logsumexp(posterior_log_probs)

    first_term = logsumexp(prior_log_probs) - np.log(M1) - np.mean(prior_log_probs)

    # Compute total information gain
    info_gain = M1*first_term + M2*second_term

    return info_gain



def compute_infogain_7(env, 
                       demos, 
                       inner_mcmc_samples_prior, 
                       inner_mcmc_samples_post,
                       outer_mcmc_samples_prior,
                       outer_mcmc_samples_post,
                        beta, 
                        log_prob_func):
    """
    Compute information gain between posterior and prior MCMC samples for demonstrations.
    
    Args:
        env: The GridWorld environment.
        demos: List of demonstrations (state-action pairs or preferences).
        mcmc_samples_1: MCMC samples from prior \( \Theta_{n-1} \).
        mcmc_samples_2: MCMC samples from posterior \( \Theta_n \).
        beta: Rationality parameter.
        log_prob_func: Function to compute log probability (either log_prob_demo or log_prob_comparison).
    
    Returns:
        float: Information gain value.
    """
    M1 = len(outer_mcmc_samples_prior)  # Number of prior samples
    M2 = len(outer_mcmc_samples_post)  # Number of posterior samples

    M1_n = len(inner_mcmc_samples_prior)
    M2_n = len(inner_mcmc_samples_post)

    # Handle initial condition (n=1)
    if len(demos) == 1:
        posterior_log_probs_outer = np.exp(np.array([log_prob_func(env, demos, theta, beta) for theta in outer_mcmc_samples_post]))
        posterior_log_probs_inner = np.exp(np.array([log_prob_func(env, demos, theta, beta) for theta in inner_mcmc_samples_post]))
        
        second_term = - np.log(np.sum(posterior_log_probs_inner)) + np.log(M2_n) + np.mean(posterior_log_probs_outer)
        
        ## Apply log on each component of posterior_log_probs
        ## Sum them and take average
        #second_term = np.mean(np.log(posterior_log_probs)) + np.log(M2) - np.log(np.sum(posterior_log_probs))
        
        #posterior_denominator = logsumexp(posterior_log_probs)

        #second_term = np.mean(posterior_log_probs - posterior_denominator + np.log(M2))
        return second_term

    # Compute log probabilities for prior and posterior samples
    posterior_log_probs_outer = np.exp(np.array([log_prob_func(env, demos, theta, beta) for theta in outer_mcmc_samples_post]))
    posterior_log_probs_inner = np.exp(np.array([log_prob_func(env, demos, theta, beta) for theta in inner_mcmc_samples_post]))
        
    prior_log_probs_outer = np.exp(np.array([log_prob_func(env, demos, theta, beta) for theta in outer_mcmc_samples_prior]))
    prior_log_probs_inner = np.exp(np.array([log_prob_func(env, demos, theta, beta) for theta in inner_mcmc_samples_prior]))
    
    
    second_term = - np.log(np.sum(posterior_log_probs_inner)) + np.log(M2_n) + np.mean(posterior_log_probs_outer)
        
    first_term = np.log((np.sum(prior_log_probs_inner))) - np.log(M1_n) - np.mean(prior_log_probs_outer)

    # Compute total information gain
    info_gain = first_term + second_term

    return info_gain


def compute_norm_infogain(env, 
                       demos, 
                       inner_mcmc_samples_prior, 
                       inner_mcmc_samples_post,
                       outer_mcmc_samples_prior,
                       outer_mcmc_samples_post,
                        beta, 
                        log_prob_func):
    """
    Compute information gain between posterior and prior MCMC samples for demonstrations.
    
    Args:
        env: The GridWorld environment.
        demos: List of demonstrations (state-action pairs or preferences).
        mcmc_samples_1: MCMC samples from prior \( \Theta_{n-1} \).
        mcmc_samples_2: MCMC samples from posterior \( \Theta_n \).
        beta: Rationality parameter.
        log_prob_func: Function to compute log probability (either log_prob_demo or log_prob_comparison).
    
    Returns:
        float: Information gain value.
    """
    M1 = len(outer_mcmc_samples_prior)  # Number of prior samples
    M2 = len(outer_mcmc_samples_post)  # Number of posterior samples

    M1_n = len(inner_mcmc_samples_prior)
    M2_n = len(inner_mcmc_samples_post)

    # Handle initial condition (n=1)
    if len(demos) == 1:
        posterior_log_probs_outer = np.exp(np.array([log_prob_func(env, demos, theta, beta) for theta in outer_mcmc_samples_post]))
        posterior_log_probs_inner = np.exp(np.array([log_prob_func(env, demos, theta, beta) for theta in inner_mcmc_samples_post]))
        
        second_term = - np.log(np.sum(posterior_log_probs_inner)) + np.log(M2_n) + np.mean(posterior_log_probs_outer)
        
        ## Apply log on each component of posterior_log_probs
        ## Sum them and take average
        #second_term = np.mean(np.log(posterior_log_probs)) + np.log(M2) - np.log(np.sum(posterior_log_probs))
        
        #posterior_denominator = logsumexp(posterior_log_probs)

        #second_term = np.mean(posterior_log_probs - posterior_denominator + np.log(M2))
        return second_term

    # Compute log probabilities for prior and posterior samples
    posterior_log_probs_outer = np.exp(np.array([log_prob_func(env, demos, theta, beta) for theta in outer_mcmc_samples_post]))
    posterior_log_probs_inner = np.exp(np.array([log_prob_func(env, demos, theta, beta) for theta in inner_mcmc_samples_post]))
        
    prior_log_probs_outer = np.exp(np.array([log_prob_func(env, demos, theta, beta) for theta in outer_mcmc_samples_prior]))
    prior_log_probs_inner = np.exp(np.array([log_prob_func(env, demos, theta, beta) for theta in inner_mcmc_samples_prior]))
    
    
    second_term = - np.log(np.sum(posterior_log_probs_inner)) + np.log(M2_n) + np.mean(posterior_log_probs_outer)
        
    first_term = np.log((np.sum(prior_log_probs_inner))) - np.log(M1_n) - np.mean(prior_log_probs_outer)

    # Compute total information gain
    info_gain = (first_term + second_term)/(first_term + np.log(M1))

    return info_gain


def compute_entropy(env,
            demos,
            inner_mcmc_samples_post,
            outer_mcmc_samples_post,
            beta,
            log_prob_func):

    M2_n = len(inner_mcmc_samples_post)
    M2 = len(outer_mcmc_samples_post)

    posterior_log_probs_outer = np.exp(np.array([log_prob_func(env, demos, theta, beta) for theta in outer_mcmc_samples_post]))
    posterior_log_probs_inner = np.exp(np.array([log_prob_func(env, demos, theta, beta) for theta in inner_mcmc_samples_post]))
        
    entropy = np.log(np.sum(posterior_log_probs_inner)) - np.log(M2_n) - np.mean(posterior_log_probs_outer) + np.log(M2)

    return entropy


import numpy as np
from scipy.special import logsumexp
from multiprocessing import Pool

def compute_entropy_parallel(env, demos, inner_mcmc_samples_post, outer_mcmc_samples_post, beta, log_prob_func):
    """
    Compute entropy using parallel processing for log probability computation.
    """

    M2_n = len(inner_mcmc_samples_post)
    M2 = len(outer_mcmc_samples_post)

    # Define a helper function to compute log probabilities
    def compute_log_prob(theta):
        return log_prob_func(env, demos, theta, beta)

    # Use multiprocessing to parallelize computations
    with Pool() as pool:
        posterior_log_probs_outer = np.exp(np.array(pool.map(compute_log_prob, outer_mcmc_samples_post)))
        posterior_log_probs_inner = np.exp(np.array(pool.map(compute_log_prob, inner_mcmc_samples_post)))

    # Compute entropy
    entropy = np.log(np.sum(posterior_log_probs_inner)) - np.log(M2_n) - np.mean(posterior_log_probs_outer) + np.log(M2)

    return entropy






def log_prob_demo(env, demos, theta, beta):
    """
    Computes the log probability of a set of demonstrations given a reward function.

    Args:
        env: The GridWorld environment.
        demos: A list of (state, action) pairs representing demonstrations.
        theta: The reward function parameters.
        beta: The rationality parameter for the likelihood model.

    Returns:
        float: The log-likelihood of the demonstrations given the reward function.
    """
    env.set_feature_weights(theta)
    q_values = ValueIteration(env).get_q_values()

    log_sum = 0.0
    for s, a in demos:
        if s not in env.terminal_states:
            log_sum += beta * q_values[s][a] - logsumexp(beta * q_values[s])

    return log_sum


def log_prob_estop(env, demos, theta, beta):
    
    env.set_feature_weights(theta)

    # Initialize the log prior as 0, assuming an uninformative prior
    log_prior = 0.0
    log_sum = log_prior  # Start the log sum with the log prior value

    for estop in demos:
        # Unpack the trajectory and stopping point
        trajectory, t = estop
        traj_len = len(trajectory)

        # Compute the cumulative reward up to the stopping point t
        reward_up_to_t = sum(env.compute_reward(s) for s, _ in trajectory[:t+1])

        # Add repeated rewards for the last step at time t until the trajectory horizon
        reward_up_to_t += (traj_len - t - 1) * env.compute_reward(trajectory[t][0])

        # Numerator: P(off | r, C) -> exp(beta * reward_up_to_t)
        stop_prob_numerator = beta * reward_up_to_t

        # reward of the whole trajectory
        traj_reward = sum(env.compute_reward(s) for s, _ in trajectory[:])
        
        #denominator = np.exp(self.beta * traj_reward) + np.exp(stop_prob_numerator)
    
        # Use the Log-Sum-Exp trick for the denominator
        max_reward = max(beta * traj_reward, stop_prob_numerator)
        log_denominator = max_reward + np.log(
            np.exp(beta * traj_reward - max_reward) +
            np.exp(stop_prob_numerator - max_reward)
        )

        # Add the log probability to the log sum
        log_sum += stop_prob_numerator - log_denominator


        # Add the log probability to the log sum
        #log_sum += stop_prob_numerator - np.log(denominator)
    
    return log_sum

def log_prob_comparison(env, demos, theta, beta):
    """
    Computes the log probability of preference demonstrations using a Boltzmann distribution.

    Args:
        env: The GridWorld environment.
        demos: A list of (trajectory1, trajectory2) pairs representing preference demonstrations.
        theta: The reward function parameters.
        beta: The rationality parameter for the likelihood model.

    Returns:
        float: The log-likelihood of the preferences given the reward function.
    """
    env.set_feature_weights(theta)

    log_sum = 0.0
    for traj1, traj2 in demos:
        reward1 = compute_reward_for_trajectory(env, traj1)
        reward2 = compute_reward_for_trajectory(env, traj2)
        log_sum += beta * reward1 - logsumexp([beta * reward1, beta * reward2])

    return log_sum
