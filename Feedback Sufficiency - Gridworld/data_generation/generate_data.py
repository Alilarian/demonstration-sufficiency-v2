"""
Some ideas for future debug

In generate_optimal_demo, do I need to multiply discount factor when I summing rewards?

"""
import sys
import os
import time
import yaml
import numpy as np
import random
from scipy.special import logsumexp

# Get current and parent directory to handle import paths
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from agent.q_learning_agent import ValueIteration

class GridWorldMDPDataGenerator:
    def __init__(self, env, q_values=None, seed=None):
        """
        Initializes the generator with the environment.
        :param env: Markov decision process environment.
        """
        self.env = env
        self.q_values = q_values

        if seed is not None:
            self._set_random_seed(seed=seed)

    def generate_optimal_demo(self, num_trajs, start_states=None):
        """
        Generates multiple optimal demonstrations consisting of state-action pairs (s, a),
        and computes the cumulative reward for each trajectory.

        :param num_trajs: Number of trajectories to generate.
        :param start_states: Optional list of starting states. If not provided, random non-terminal states will be used.
        :return: A list of tuples where each tuple contains an optimal trajectory and its associated cumulative reward.
        """

        trajectories_with_rewards = []

        # Get all non-terminal states
        non_terminal_states = [s for s in range(self.env.get_num_states()) 
                            if s not in self.env.terminal_states]

        # Handle start states: If not provided, randomly select unique non-terminal starting states
        if start_states is None:
            if num_trajs > len(non_terminal_states):
                raise ValueError("Number of trajectories exceeds the number of available non-terminal states.")
            start_states = np.random.choice(non_terminal_states, size=num_trajs, replace=False)

        for current_state in start_states:
            max_traj_length = self.env.get_num_states()
            optimal_trajectory = []
            cumulative_reward = 0  # Initialize cumulative reward for the trajectory

            # Generate the trajectory until a terminal state is reached or max length is reached
            while current_state not in self.env.terminal_states and len(optimal_trajectory) < max_traj_length:
                # Generate an optimal action, breaking ties uniformly at random
                act = np.random.choice(self.arg_max_set(self.q_values[current_state]))
                optimal_trajectory.append((current_state, act))

                # Compute reward for the current state
                reward = self.env.compute_reward(current_state)
                cumulative_reward += reward

                # Sample the next state based on transition probabilities
                probs = self.env.transitions[current_state][act]
                next_state = np.random.choice(self.env.num_states, p=probs)
                current_state = next_state

            # Handle the last state if it's terminal
            if current_state in self.env.terminal_states:
                reward = self.env.compute_reward(current_state)
                cumulative_reward += reward
                # Append the terminal state with a dummy action (-1 or None) if needed
                optimal_trajectory.append((current_state, None))  # Terminal state, no action


            # Store the trajectory and its cumulative reward
            trajectories_with_rewards.append((optimal_trajectory, cumulative_reward))

        return trajectories_with_rewards


    def generate_random_demo(self, num_trajs, start_states=None):
        """
        Generates multiple random trajectories consisting of state-action pairs (s, a),
        and computes the cumulative reward for each trajectory.

        :param num_trajs: Number of trajectories to generate.
        :param start_states: Optional list of starting states. If not provided, random non-terminal states will be used.
        :return: A list of tuples where each tuple contains a random trajectory and its associated cumulative reward.
        """
        trajectories_with_rewards = []

        # Get all non-terminal states
        non_terminal_states = [
            s for s in range(self.env.get_num_states()) 
            if s not in self.env.terminal_states
        ]
        # Handle start states: If not provided, randomly select unique non-terminal starting states
        start_states = np.random.choice(non_terminal_states, size=num_trajs, replace=True)

        for current_state in start_states:
            max_traj_length = self.env.get_num_states()
            random_trajectory = []
            cumulative_reward = 0  # Initialize cumulative reward for the trajectory

            # Generate the trajectory until a terminal state is reached or max length is reached
            while current_state not in self.env.terminal_states and len(random_trajectory) < max_traj_length:
                # Select a random action
                act = np.random.choice(range(self.env.num_actions))
                random_trajectory.append((current_state, act))

                # Compute reward for the current state
                reward = self.env.compute_reward(current_state)
                cumulative_reward += reward

                # Sample the next state based on transition probabilities
                probs = self.env.transitions[current_state][act]
                next_state = np.random.choice(self.env.num_states, p=probs)
                current_state = next_state

            # Handle the last state if it's terminal
            if (current_state in self.env.terminal_states or 
                    len(random_trajectory) < max_traj_length):
                
                reward = self.env.compute_reward(current_state)
                cumulative_reward += reward
                # Append the terminal state with a dummy action (-1 or None) if needed
                random_trajectory.append((current_state, None))  # Terminal state, no action

            # Store the trajectory and its cumulative reward
            trajectories_with_rewards.append((random_trajectory, cumulative_reward))

        return trajectories_with_rewards

    def generate_pairwise_comparisons(self, strategy="random_vs_random", num_trajs=10):
        """
        Generates pairwise comparisons between trajectories based on rewards.

        Strategies:
        1. 'random_vs_random' - Generate multiple random trajectories and compare all pairs.
        2. 'same_start_state' - Generate two trajectories from the same start state and compare them.

        :param strategy: The strategy to use for generating pairwise comparisons.
        :param num_trajs: Number of trajectories to generate.
        :return: List of tuples where each tuple contains a pair of trajectories for comparison.
        """
        pairwise_comparisons = []

        if strategy == "random_vs_random":
            # Generate multiple random trajectories and compare all unique pairs
            trajectories = self.generate_random_demo(num_trajs)
            for i in range(len(trajectories)):
                for j in range(len(trajectories)):
                    if i != j:
                        traj_1, reward_1 = trajectories[i]
                        traj_2, reward_2 = trajectories[j]
                        if reward_1 > reward_2:
                            pairwise_comparisons.append((traj_1, traj_2))

        elif strategy == "same_start_state":
            # Generate two random trajectories from the same start state and compare them
            non_terminal_states = [
                s for s in range(self.env.get_num_states()) if s not in self.env.terminal_states
            ]

            start_states = np.random.choice(non_terminal_states, size=num_trajs, replace=True)

            for start_state in start_states:
                traj_1, reward_1 = self.generate_random_demo(1, [start_state])[0]
                traj_2, reward_2 = self.generate_random_demo(1, [start_state])[0]

                if reward_1 > reward_2:
                    pairwise_comparisons.append((traj_1, traj_2))
                elif reward_2 > reward_1:
                    pairwise_comparisons.append((traj_2, traj_1))
                    
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

        return pairwise_comparisons

    def generate_estop(self, beta, num_trajs, start_states=None):
        """
        Generates E-stops for random trajectories using the human likelihood model.
        
        :param beta: Sensitivity parameter for human decision-making.
        :param num_trajs: Number of trajectories to generate.
        :return: List of E-stop events (trajectories with stopping point t).
        """
        # Generate random trajectories first
        trajectories_with_rewards = self.generate_random_demo(num_trajs, start_states)
        
        estop_trajectories = []

        # Iterate over each trajectory
        for trajectory, cumulative_reward in trajectories_with_rewards:
            T = len(trajectory)  # Length of the trajectory
            stop_probs = []
            
            # Calculate stop probabilities for each possible stopping point t
            for t in range(T):
                # Sub-trajectory ξ_0:t (use cumulative rewards up to point t)
                reward_up_to_t = sum(self.env.compute_reward(s) for s, _ in trajectory[:t+1])
                stop_prob_numerator = np.exp(beta * reward_up_to_t)
                
                # Compute denominator (normalization factor for the entire trajectory)
                stop_prob_denominator = sum(np.exp(beta * sum(self.env.compute_reward(s) for s, _ in trajectory[:k+1])) for k in range(T))
                
                # Calculate stop probability for stopping at time t
                stop_prob = stop_prob_numerator / stop_prob_denominator
                stop_probs.append(stop_prob)
            
            # Select stopping point t based on the calculated probabilities
            stop_point = np.random.choice(range(T), p=stop_probs)
            
            # Append the trajectory with its stopping point to the result list
            estop_trajectories.append((trajectory, stop_point))

        return estop_trajectories

    @staticmethod
    def arg_max_set(q_values):
        """
        Returns the set of actions corresponding to the maximum Q-value for a given state.
        :param q_values: Q-values for the current state.
        :return: List of actions with the maximum Q-value.
        """
        max_q = np.max(q_values)
        return np.flatnonzero(q_values == max_q)

    def _set_random_seed(self, seed):
        """
        Set the random seed for numpy, random, and any other libraries that require seeding.
        """
        np.random.seed(seed)
        random.seed(seed)

def generate_random_trajectory(env, max_horizon=25):
    """
    Generate a random trajectory of fixed length (max_horizon + 1) using random actions.
    The state is stored as an integer index (raw_index) instead of (row, col).
    
    Args:
        env: The GridWorld environment.
        max_horizon (int): Maximum length of the trajectory.
        
    Returns:
        list of (state_index, action) tuples.
    """
    trajectory = []
    obsv = env.reset()  # Reset environment and get initial observation.
    agent_position = obsv["agent"]  # [row, col]
    terminal_states = obsv["terminal states"]  # List of terminal states as indices

    # Compute the raw index (integer) for the initial state.
    
    try:
        state = agent_position[0] * env.columns + agent_position[1]
    except:
        state = agent_position[0] * env.size + agent_position[1]

    for step in range(max_horizon):
        # Append the current state and chosen action
        if state in terminal_states:
            # Append terminal state with None action to indicate stopping
            trajectory.append((state, None))
            break  # Stop generating the trajectory if a terminal state is reached.

        # Choose a random action uniformly.
        action = np.random.choice(env.num_actions)

        # Sample the next state based on transition probabilities.
        next_state = np.random.choice(env.num_states, p=env.transitions[state][action])

        # Append (current state, chosen action) to the trajectory.
        trajectory.append((state, action))

        # Update state (now directly using raw index).
        state = next_state

    # Ensure the terminal state is appended if the loop exits without adding it
    #if state in terminal_states and (len(trajectory) == 0 or trajectory[-1][0] != state):
    #    trajectory.append((state, None))  # Append terminal state explicitly

    return trajectory

def generate_pairwise_comparisons(env, num_trajs=10, max_horizon=25, num_comparisons=10):
    """
    Generates a fixed number of pairwise comparisons between randomly generated trajectories.

    Args:
        env: The GridWorld environment.
        num_trajs (int): Number of trajectories to generate.
        max_horizon (int): Maximum length of each trajectory.
        num_comparisons (int): Number of comparisons to return.

    Returns:
        List of tuples containing a pair of trajectories for comparison.
    """
    pairwise_comparisons = []
    trajectories = []

    # Generate random trajectories
    for _ in range(num_trajs):
        traj = generate_random_trajectory(env, max_horizon)
        total_reward = sum(env.compute_reward(state) for state, action in traj)  # Ignore terminal state
        trajectories.append((traj, total_reward))

    # Compare all unique trajectory pairs
    for i in range(len(trajectories)):
        for j in range(i + 1, len(trajectories)):  # Avoid duplicate comparisons
            traj_1, reward_1 = trajectories[i]
            traj_2, reward_2 = trajectories[j]

            if reward_1 > reward_2:
                pairwise_comparisons.append((traj_1, traj_2))
            elif reward_2 > reward_1:
                pairwise_comparisons.append((traj_2, traj_1))

    # Ensure we return exactly `num_comparisons` comparisons
    return random.sample(pairwise_comparisons, min(num_comparisons, len(pairwise_comparisons)))













def generate_random_trajectory_diff_start(env, max_horizon=25):
    """
    Generate a random trajectory of up to max_horizon steps using random actions.
    The trajectory starts from a random non-terminal state and may or may not reach the goal.

    Args:
        env: The GridWorld environment.
        max_horizon (int): Maximum length of the trajectory.

    Returns:
        list of (state_index, action) tuples.
    """
    trajectory = []

    # Get all non-terminal states except goal
    non_terminal_states = [s for s in range(env.num_states) if s not in env.terminal_states]
    #non_terminal_states = [0, 3]

    # Select a random start state (not the goal)
    start_state = random.choice(non_terminal_states)
    state = start_state

    for step in range(max_horizon):
        # Append current state and action
        if state in env.terminal_states:
            trajectory.append((state, None))  # Terminal state reached
            break

        # Choose a random action uniformly
        action = np.random.choice(env.num_actions)

        # Sample the next state based on transition probabilities
        next_state = np.random.choice(env.num_states, p=env.transitions[state][action])

        # Append (current state, chosen action) to the trajectory
        trajectory.append((state, action))

        # Update state
        state = next_state

    return trajectory





def simulate_human_estop(env, full_trajectory, beta=2.0, gamma=1.0, fixed_length=None):
    """
    Simulates human E-stop (early stopping) behavior in a GridWorld environment and ensures all output trajectories have the same length.

    Args:
        env (NoisyLinearRewardFeaturizedGridWorldEnv): The environment instance.
        full_trajectory (list): A full-length trajectory as [(state, action), ...].
        beta (float): Sensitivity parameter for Boltzmann distribution.
        gamma (float): Discount factor for cumulative rewards.
        fixed_length (int, optional): Desired fixed length for the output trajectory. If the trajectory is shorter, the last step is repeated.

    Returns:
        tuple: (trajectory, stopping_time)
    """
    cumulative_rewards = []
    current_reward = 0

    for k, (state, _) in enumerate(full_trajectory):
        if state is None:  # Handle padding
            break

        # Compute reward for the current state using the environment function
        reward = env.compute_reward(state)  # Now using the built-in reward function

        # Discounted cumulative reward up to step k
        current_reward += (gamma**k) * reward
        cumulative_rewards.append(current_reward)

    # Convert to numpy array for stable computation
    cumulative_rewards = np.array(cumulative_rewards)

    # Use `logsumexp` for numerical stability
    log_denominator = logsumexp(beta * cumulative_rewards)
    log_numerator = beta * cumulative_rewards  # Log of exp(beta * cumulative_rewards)

    # Compute stopping probabilities in log-space
    log_probabilities = log_numerator - log_denominator
    probabilities = np.exp(log_probabilities)  # Convert back to normal probability values

    # Select stopping time using highest probability
    t_stop = np.argmax(probabilities)

    # Pad the trajectory to ensure it matches the fixed length
    if fixed_length is not None:
        last_step = full_trajectory[-1]
        while len(full_trajectory) < fixed_length:
            full_trajectory.append(last_step)

    return (full_trajectory[:fixed_length] if fixed_length else full_trajectory, t_stop)

def simulate_human_estop_v2(env, full_trajectory, beta=2.0, gamma=1.0):
    """
    Simulates E-stop data based on the provided likelihood model.

    Args:
        env (NoisyLinearRewardFeaturizedGridWorldEnv): The environment instance.
        full_trajectory (list): A full-length trajectory as [(state, action), ...].
        beta (float): Sensitivity parameter for Boltzmann distribution.
        gamma (float): Discount factor for cumulative rewards.

    Returns:
        tuple: (trajectory, stopping_time)
    """
    traj_len = len(full_trajectory)

    # Compute cumulative reward for the entire trajectory
    traj_reward = sum(env.compute_reward(s) for s, _ in full_trajectory)

    # Initialize variables
    cumulative_rewards = []
    probabilities = []

    # Compute cumulative rewards up to each time step and probabilities
    for t in range(traj_len):
        # Reward up to time t
        reward_up_to_t = sum(env.compute_reward(s) for s, _ in full_trajectory[:t+1])

        # Add repeated reward for the last step
        reward_up_to_t += (traj_len - t - 1) * env.compute_reward(full_trajectory[t][0])

        # Numerator and denominator for the stopping probability
        numerator = beta * reward_up_to_t
        denominator = logsumexp([beta * traj_reward, numerator])
        

        # Compute the probability of stopping at time t
        stop_probability = numerator - denominator
        probabilities.append(stop_probability)

    # Normalize probabilities (to ensure numerical stability)
    probabilities = np.array(probabilities)
    #probabilities /= probabilities.sum()

    # Sample stopping point t_stop from the computed probabilities
    #t_stop = np.random.choice(len(probabilities), p=probabilities)
    t_stop = np.argmax(probabilities)

    # Return the trajectory and the stopping point
    return (full_trajectory, t_stop)