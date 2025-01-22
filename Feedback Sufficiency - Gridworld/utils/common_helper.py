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

def compute_infogain(env, demo, mcmc_samples, beta):
    """
    Compute the information gain as shown in the equation.

    Args:
        env: The environment object, which allows setting feature weights and computing Q-values.
        demo: A tuple (s, a) representing the demonstration (state, action).
        mcmc_samples: A list of sampled theta values (feature weights).
        beta: Inverse temperature parameter for the Boltzmann distribution.

    Returns:
        Information gain (float).
    """
    M = len(mcmc_samples)  # Number of MCMC samples
    main_sum = 0  # To accumulate the main sum

    s, a = demo  # Unpack the state and action from the demo
    #print("State and action - Inside compute_infogain")
    #print(s)
    #print(a)

    # Step 1: Compute P(Dn | theta) for all thetas
    p_dn_given_theta_list = []  # Store P(Dn | theta) for each theta

    for theta in mcmc_samples:
        # Set feature weights to the current theta
        env.set_feature_weights(theta)

        # Perform value iteration to get Q-values
        val_iter = ValueIteration(env)
        q_values = val_iter.get_q_values()

        # Compute P(Dn | theta) using the Boltzmann distribution
        p_dn_given_theta = np.exp(beta * q_values[s][a]) / logsumexp(beta * q_values[s])
        p_dn_given_theta_list.append(p_dn_given_theta)

    # Step 2: Compute the marginal likelihood P(Dn)
    p_dn = np.mean(p_dn_given_theta_list)  # Marginal likelihood

    # Step 3: Compute the main sum
    for i, theta in enumerate(mcmc_samples):
        p_dn_given_theta = p_dn_given_theta_list[i]

        # Compute the first log term
        log_term1 = np.log(M * p_dn_given_theta)

        # Compute the second log term
        log_term2 = np.log(np.sum(p_dn_given_theta_list))

        # Accumulate the information gain
        main_sum += p_dn_given_theta * (log_term1 - log_term2)

    # Step 4: Normalize by the number of samples M
    info_gain = main_sum / M

    return info_gain


def compute_infogain_log(env, demos, mcmc_samples, beta):
    """
    Compute the information gain in log scale for a list of demonstrations.

    Args:
        env: The environment object, which allows setting feature weights and computing Q-values.
        demos: A list of tuples [(s, a)] representing the demonstrations (states and actions).
        mcmc_samples: A list of sampled theta values (feature weights).
        beta: Inverse temperature parameter for the Boltzmann distribution.

    Returns:
        Information gain (float).
    """
    M = len(mcmc_samples)  # Number of MCMC samples
    main_sum = 0  # To accumulate the main sum

    # Step 1: Outer loop over demonstrations
    for demo in demos:
        s, a = demo  # Unpack the state and action from the current demo

        # Step 2: Compute log P(Dn | theta) for all thetas
        log_p_dn_given_theta_list = []  # Store log P(Dn | theta) for each theta

        for theta in mcmc_samples:
            # Set feature weights to the current theta
            env.set_feature_weights(theta)

            # Perform value iteration to get Q-values
            val_iter = ValueIteration(env)
            q_values = val_iter.get_q_values()

            # Compute log P(Dn | theta) using the log-sum-exp trick
            log_p_dn_given_theta = beta * q_values[s][a] - logsumexp(beta * q_values[s])
            log_p_dn_given_theta_list.append(log_p_dn_given_theta)

        # Step 3: Compute log marginal likelihood log P(Dn)
        log_p_dn_given_theta_array = np.array(log_p_dn_given_theta_list)
        log_p_dn = logsumexp(log_p_dn_given_theta_array - np.log(M))  # Normalize by M

        # Step 4: Compute the main sum for the current demo
        for i, theta in enumerate(mcmc_samples):
            log_p_dn_given_theta = log_p_dn_given_theta_list[i]

            # Compute the first log term: log(M * P(Dn | theta)) = log(M) + log P(Dn | theta)
            log_term1 = np.log(M) + log_p_dn_given_theta

            # Compute the second log term: log P(Dn)
            log_term2 = log_p_dn

            # Convert back from log to linear scale for weighted sum
            main_sum += np.exp(log_p_dn_given_theta) * (log_term1 - log_term2)

    # Step 5: Normalize by the number of samples M and the number of demonstrations
    info_gain = main_sum / (M)

    return info_gain


def compute_infogain_log___(env, demo, mcmc_samples, beta):
    """
    Compute the information gain in log scale.

    Args:
        env: The environment object, which allows setting feature weights and computing Q-values.
        demo: A tuple (s, a) representing the demonstration (state, action).
        mcmc_samples: A list of sampled theta values (feature weights).
        beta: Inverse temperature parameter for the Boltzmann distribution.

    Returns:
        Information gain (float).
    """
    M = len(mcmc_samples)  # Number of MCMC samples
    main_sum = 0  # To accumulate the main sum

    s, a = demo  # Unpack the state and action from the demo

    # Step 1: Compute log P(Dn | theta) for all thetas
    log_p_dn_given_theta_list = []  # Store log P(Dn | theta) for each theta

    for theta in mcmc_samples:
        # Set feature weights to the current theta
        env.set_feature_weights(theta)

        # Perform value iteration to get Q-values
        val_iter = ValueIteration(env)
        q_values = val_iter.get_q_values()

        # Compute log P(Dn | theta) using the log-sum-exp trick
        log_p_dn_given_theta = beta * q_values[s][a] - logsumexp(beta * q_values[s])
        log_p_dn_given_theta_list.append(log_p_dn_given_theta)

    # Step 2: Compute log marginal likelihood log P(Dn)
    log_p_dn_given_theta_array = np.array(log_p_dn_given_theta_list)
    log_p_dn = logsumexp(log_p_dn_given_theta_array - np.log(M))  # Normalize by M

    # Step 3: Compute the main sum
    for i, theta in enumerate(mcmc_samples):
        log_p_dn_given_theta = log_p_dn_given_theta_list[i]

        # Compute the first log term: log(M * P(Dn | theta)) = log(M) + log P(Dn | theta)
        log_term1 = np.log(M) + log_p_dn_given_theta

        # Compute the second log term: log P(Dn)
        log_term2 = log_p_dn

        # Convert back from log to linear scale for weighted sum
        main_sum += np.exp(log_p_dn_given_theta) * (log_term1 - log_term2)

    # Step 4: Normalize by the number of samples M
    info_gain = main_sum / M

    return info_gain
