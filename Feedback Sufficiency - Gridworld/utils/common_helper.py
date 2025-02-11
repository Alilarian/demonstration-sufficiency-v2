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

def compute_policy_loss_avar_bound(mcmc_samples, env, map_policy, random_normalization, alpha, delta):
    """
    Computes the counterfactual policy losses and calculates the a-VaR (Value at Risk) bound.

    Args:
        mcmc_samples (list): List of MCMC sampled rewards from the BIRL process.
        env: The environment object.
        map_policy: The MAP (Maximum a Posteriori) policy from BIRL.
        random_normalization (bool): Whether to normalize using a random policy.
        alpha (float): Confidence level parameter.
        delta (float): Risk level parameter.

    Returns:
        float: The computed a-VaR bound.
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

    # Step 3: Compute the VaR (Value at Risk) bound
    N_burned = len(mcmc_samples)
    k = math.ceil(N_burned * alpha + norm.ppf(1 - delta) * np.sqrt(N_burned * alpha * (1 - alpha)) - 0.5)
    k = min(k, N_burned - 1)  # Ensure k doesn't exceed the number of samples

    # Return the computed a-VaR bound
    return policy_losses[k]

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

def compute_infogain(env, demos, mcmc_samples_1, mcmc_samples_2, beta):
    """
    Compute information gain between posterior and prior MCMC samples for demonstrations.

    Args:
        env: The GridWorld environment.
        demos: List of demonstrations (state-action pairs).
        mcmc_samples_1: MCMC samples from prior \( \Theta_{n-1} \).
        mcmc_samples_2: MCMC samples from posterior \( \Theta_n \).
        beta: Rationality parameter.

    Returns:
        float: Information gain value.
    """

    M2 = len(mcmc_samples_2)  # Number of posterior samples

    # Handle initial condition (n=1)
    if len(demos) == 1:
        # Only compute the second term
        #posterior_denominator = sum(compute_log_prob(env, demos, theta, beta) for theta in mcmc_samples_2)
        posterior_denominator = logsumexp(
            [compute_log_prob(env, demos, theta, beta) for theta in mcmc_samples_2]
        )

        print("posterior_denominator: ", posterior_denominator)

        second_term = 0
        for theta_posterior in mcmc_samples_2:
            log_p_demos_posterior = compute_log_prob(env, demos, theta_posterior, beta)
            second_term += log_p_demos_posterior - posterior_denominator + np.log(M2)
        second_term /= M2

        return second_term

    # General case for n > 1
    M1 = len(mcmc_samples_1)  # Number of prior samples

    # Precompute normalization term for prior
    #prior_denominator = sum(compute_log_prob(env, demos[:-1], theta, beta) for theta in mcmc_samples_1)
    prior_denominator = logsumexp(
        [compute_log_prob(env, demos[:-1], theta, beta) for theta in mcmc_samples_1]
    )


    first_term = 0
    for theta_prior in mcmc_samples_1:
        log_p_demos_prior = compute_log_prob(env, demos[:-1], theta_prior, beta)
        first_term += prior_denominator - log_p_demos_prior - np.log(M1)
    first_term /= M1

    # Precompute normalization term for posterior
    #posterior_denominator = sum(compute_log_prob(env, demos, theta, beta) for theta in mcmc_samples_2)
    posterior_denominator = logsumexp(
            [compute_log_prob(env, demos, theta, beta) for theta in mcmc_samples_2]
        )

    second_term = 0
    for theta_posterior in mcmc_samples_2:
        log_p_demos_posterior = compute_log_prob(env, demos, theta_posterior, beta)
        second_term += log_p_demos_posterior - posterior_denominator + np.log(M2)
    second_term /= M2

    # Compute total information gain
    info_gain = first_term + second_term

    return info_gain

def compute_log_prob(env, demos, theta, beta):
    """
    Compute the log probability of the demonstrations given theta.

    Args:
        env: The GridWorld environment.
        demos: List of demonstrations (state-action pairs).
        theta: Current reward weights (parameter vector).
        beta: Rationality parameter.

    Returns:
        float: Log probability.
    """
    
    env.set_feature_weights(theta)

    val_iter = ValueIteration(env)

    #if self.env in self.value_iters:
        
    #    q_values = calculate_q_values(self.env, V = self.value_iters[self.env], epsilon = self.epsilon)
    #else:
    #q_values = calculate_q_values(self.env, storage = self.value_iters, epsilon = self.epsilon)
    q_values = val_iter.get_q_values()
    #calculate the log likelihood of the reward hypothesis given the demonstrations
    log_prior = 0.0  #assume unimformative prior
    log_sum = log_prior
    for s, a in demos:
        if (s not in env.terminal_states):  # there are no counterfactuals in a terminal state

            Z_exponents = beta * q_values[s]
            log_sum += beta * q_values[s][a] - logsumexp(Z_exponents)
            
    return log_sum