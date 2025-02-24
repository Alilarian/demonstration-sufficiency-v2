import gymnasium as gym
from gym import spaces
import numpy as np
import random
import pygame

class NoisyLinearRewardFeaturizedGridWorldEnv(gym.Env):
    """
    A custom GridWorld environment with noisy transitions and linear rewards based on feature vectors.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, gamma, render_mode=None, size=5, noise_prob=0.1, terminal_states=None):
        super(NoisyLinearRewardFeaturizedGridWorldEnv, self).__init__()
        self.size = size  # Size of the grid
        self.window_size = 512  # Window size for rendering
        self.noise_prob = noise_prob  # Noise probability for actions
        self.gamma = gamma  # Discount factor of MDP

        # Define feature vectors for each color in the grid
        #self.colors_to_features = {
        #    "blue": np.array([1, 0, 0, 0, 0, 0, 0]),
        #    "red": np.array([0, 1, 0, 0, 0, 0, 0]),
        #    "green": np.array([0, 0, 1, 0, 0, 0, 0]),
        #    "yellow": np.array([0, 0, 0, 1, 0, 0, 0]),
        #    "purple": np.array([0, 0, 0, 0, 1, 0, 0]),
        #    "orange": np.array([0, 0, 0, 0, 0, 1, 0]),
        #    "black": np.array([0, 0, 0, 0, 0, 0, 1])
        #}
        self.colors_to_features = {
            "blue": np.array([1, 0, 0, 0]),
            "red": np.array([0, 1, 0, 0]),
            "green": np.array([0, 0, 1, 0]),
            "black": np.array([0, 0, 0, 1]),
        }

        self.set_random_seed(42)
        # Define the observation and action spaces
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        })
        self.action_space = spaces.Discrete(4)
        # Initialize transition matrix
        self.num_states = self.get_num_states()
        self.num_actions = self.get_num_actions()
        # Allow for multiple terminal states
        if terminal_states is None:
            self.terminal_states = [self.num_states - 1]  # Default terminal state (bottom-right)
        else:
            self.terminal_states = terminal_states  # Set multiple terminal states

        self.num_feat = len(self.colors_to_features['blue'])
        self.grid_features = self._initialize_grid_features(size)

        # Linear weight vector for calculating rewards
        self.feature_weights = sorted(np.random.randn(self.num_feat))

        self.transitions = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.initialize_transition_matrix()
        
        self._set_terminal_state_transitions()  # Handle terminal states

        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # Validate the render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]

    def initialize_transition_matrix(self):
        """
        Initializes the transition matrix with noisy state transitions.
        """
        RIGHT = 3
        UP = 0
        LEFT = 2
        DOWN = 1
        num_states = self.size * self.size

        for s in range(num_states):
            row, col = divmod(s, self.size)

            # Transitions for UP
            if row > 0:
                self.transitions[s][UP][s - self.size] = 1.0 - (2 * self.noise_prob)
            else:
                self.transitions[s][UP][s] = 1.0 - (2 * self.noise_prob)
            if col > 0:
                self.transitions[s][UP][s - 1] = self.noise_prob
            else:
                self.transitions[s][UP][s] = self.noise_prob
            if col < self.size - 1:
                self.transitions[s][UP][s + 1] = self.noise_prob
            else:
                self.transitions[s][UP][s] = self.noise_prob

            # Handle top-left and top-right corners for UP
            if s < self.size and col == 0:  # Top-left corner
                self.transitions[s][UP][s] = 1.0 - self.noise_prob
            elif s < self.size and col == self.size - 1:  # Top-right corner
                self.transitions[s][UP][s] = 1.0 - self.noise_prob
        
        #for s in range(num_states):
        #    row, col = divmod(s, self.size)
            # Transitions for DOWN
            if row < self.size - 1:
                self.transitions[s][DOWN][s + self.size] = 1.0 - (2 * self.noise_prob)
            else:
                self.transitions[s][DOWN][s] = 1.0 - (2 * self.noise_prob)
            if col > 0:
                self.transitions[s][DOWN][s - 1] = self.noise_prob
            else:
                self.transitions[s][DOWN][s] = self.noise_prob
            if col < self.size - 1:
                self.transitions[s][DOWN][s + 1] = self.noise_prob
            else:
                self.transitions[s][DOWN][s] = self.noise_prob

            # Handle bottom-left and bottom-right corners for DOWN
            if s >= (self.size - 1) * self.size and col == 0:  # Bottom-left corner
                self.transitions[s][DOWN][s] = 1.0 - self.noise_prob
            elif s >= (self.size - 1) * self.size and col == self.size - 1:  # Bottom-right corner
                self.transitions[s][DOWN][s] = 1.0 - self.noise_prob
        
        #for s in range(num_states):
        #    row, col = divmod(s, self.size)
            # Transitions for LEFT
            if col > 0:
                self.transitions[s][LEFT][s - 1] = 1.0 - (2 * self.noise_prob)
            else:
                self.transitions[s][LEFT][s] = 1.0 - (2 * self.noise_prob)
            if row > 0:
                self.transitions[s][LEFT][s - self.size] = self.noise_prob
            else:
                self.transitions[s][LEFT][s] = self.noise_prob
            if row < self.size - 1:
                self.transitions[s][LEFT][s + self.size] = self.noise_prob
            else:
                self.transitions[s][LEFT][s] = self.noise_prob

            # Handle top-left and bottom-left corners for LEFT
            if s < self.size and col == 0:  # Top-left corner
                self.transitions[s][LEFT][s] = 1.0 - self.noise_prob
            elif s >= (self.size - 1) * self.size and col == 0:  # Bottom-left corner
                self.transitions[s][LEFT][s] = 1.0 - self.noise_prob

        #for s in range(num_states):
        #    row, col = divmod(s, self.size)
            # Transitions for RIGHT
            if col < self.size - 1:
                self.transitions[s][RIGHT][s + 1] = 1.0 - (2 * self.noise_prob)
            else:
                self.transitions[s][RIGHT][s] = 1.0 - (2 * self.noise_prob)
            if row > 0:
                self.transitions[s][RIGHT][s - self.size] = self.noise_prob
            else:
                self.transitions[s][RIGHT][s] = self.noise_prob
            if row < self.size - 1:
                self.transitions[s][RIGHT][s + self.size] = self.noise_prob
            else:
                self.transitions[s][RIGHT][s] = self.noise_prob

            # Handle top-right and bottom-right corners for RIGHT
            if s < self.size and col == self.size - 1:  # Top-right corner
                self.transitions[s][RIGHT][s] = 1.0 - self.noise_prob
            elif s >= (self.size - 1) * self.size and col == self.size - 1:  # Bottom-right corner
                self.transitions[s][RIGHT][s] = 1.0 - self.noise_prob

    def _initialize_grid_features(self, size):
        """
        Initializes the grid features with random colors, excluding 'black' for the terminal cells.
        """
        available_colors = [color for color in self.colors_to_features.keys() if color != "black"]
        grid_features = np.random.choice(available_colors, size=(size, size))
        for terminal_state in self.terminal_states:
            row, col = divmod(terminal_state, self.size)
            grid_features[row, col] = "black"  # Set terminal states to black
        return grid_features

    def _set_terminal_state_transitions(self):
        """
        Sets the transition behavior for terminal states (self-loops for all terminal states).
        """
        for terminal_state in self.terminal_states:
            self.transitions[terminal_state, :, :] = 0
            self.transitions[terminal_state, :, terminal_state] = 1

    def step(self, action):
        """
        Executes the given action and updates the agent's position with noisy transitions.
        """
        row, col = self._agent_location
        raw_index = row * self.size + col

        # Sample the next state based on transition probabilities
        next_state = random.choices(range(self.size * self.size), self.transitions[raw_index][action])[0]
        new_row, new_col = divmod(next_state, self.size)

        # Check if the agent stayed in the same cell
        if next_state == raw_index:
            reward = 0  # Assign a reward of 0 if the agent stays in the same state
        else:
            reward = self.compute_reward(next_state)

        # Update the agent's position
        self._agent_location = np.array([new_row, new_col])

        # Check if we reached a terminal state
        terminated = next_state in self.terminal_states

        observation = self.get_observation()

        if self.render_mode == "human":
            self.render_grid_frame()

        return observation, reward, terminated, False

    def reset(self, seed=None, options=None):
        """
        Resets the environment to the initial state and returns the initial observation.
        """
        self.set_random_seed(seed)
        super().reset(seed=seed)

        # Randomly initialize the agent's position, making sure it's not in a terminal state
        while True:
            self._agent_location = self.np_random.integers(0, self.size - 1, size=2)
            raw_index = self._agent_location[0] * self.size + self._agent_location[1]
            if raw_index not in self.terminal_states:
                break

        observation = self.get_observation()
        #info = self.get_additional_info()

        if self.render_mode == "human":
            self.render_grid_frame()

        return observation

    def get_observation(self):
        """Returns the current observation (agent and target positions)."""
        return {"agent": self._agent_location, "terminal states": self.terminal_states}

    def compute_reward(self, state):
        """
        Computes the reward for a given state based on its feature vector and the feature weights.
        """
        row, col = divmod(state, self.size)
        cell_features = self.get_cell_features([row, col])
        return np.dot(cell_features, self.feature_weights)

    def get_cell_features(self, position):
        """
        Returns the feature vector of the grid cell at the given position.
        """
        color = self.grid_features[position[0], position[1]]
        return self.colors_to_features[color]

    def render(self):
        """Renders the environment in 'human' or 'rgb_array' mode."""
        if self.render_mode == "rgb_array":
            return self.render_grid_frame()
        elif self.render_mode == "human":
            self.render_grid_frame()

    def render_grid_frame(self):
        """Renders the grid and agent in PyGame."""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size  # Size of each grid square in pixels

        self._draw_grid(canvas, pix_square_size)
        self._draw_agent(canvas, pix_square_size)
        self._draw_gridlines(canvas, pix_square_size)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def _draw_grid(self, canvas, pix_square_size):
        """Draws the grid and the target cell on the canvas."""
        color_map = {
            "blue": (0, 0, 255),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "yellow": (255, 255, 0),
        }
        # Draw the target cells as black
        for terminal_state in self.terminal_states:
            row, col = divmod(terminal_state, self.size)
            pygame.draw.rect(canvas, (0, 0, 0), pygame.Rect(pix_square_size * np.array([row, col]), (pix_square_size, pix_square_size)))

        # Draw the colored grid cells
        for x in range(self.size):
            for y in range(self.size):
                if any([x == t_row and y == t_col for t_row, t_col in [divmod(ts, self.size) for ts in self.terminal_states]]):
                    continue
                color = color_map[self.grid_features[x, y]]
                pygame.draw.rect(canvas, color, pygame.Rect(pix_square_size * np.array([x, y]), (pix_square_size, pix_square_size)))

    def _draw_agent(self, canvas, pix_square_size):
        """Draws the agent as a circle on the grid."""
        pygame.draw.circle(canvas, (42, 42, 42), (self._agent_location + 0.5) * pix_square_size, pix_square_size / 3)

    def _draw_gridlines(self, canvas, pix_square_size):
        """Draws the gridlines on the canvas."""
        for x in range(self.size + 1):
            pygame.draw.line(canvas, 0, (0, pix_square_size * x), (self.window_size, pix_square_size * x), width=3)
            pygame.draw.line(canvas, 0, (pix_square_size * x, 0), (pix_square_size * x, self.window_size), width=3)

    def set_random_seed(self, seed):

        """Sets the random seed for reproducibility."""
        #np.random.seed(seed)
        random.seed(seed)

    def get_discount_factor(self):
        return self.gamma
    
    def get_num_actions(self):
        return self.action_space.n
    
    def get_num_states(self):
        return self.size * self.size

    def get_feature_weights(self):
        return self.feature_weights
    
    def set_feature_weights(self, weights):
        """Set and normalize a new weight vector for feature-based rewards."""
        self.feature_weights = weights / np.linalg.norm(weights)

    def compute_reward(self, state):
        """
        Computes the reward for a given state based on its feature vector and the feature weights.
        """
        row, col = divmod(state, self.size)
        cell_features = self.get_cell_features([row, col])
        return np.dot(cell_features, self.feature_weights)

    def get_cell_features(self, position):

        """
        Returns the feature vector of the grid cell at the given position.
        """
        color = self.grid_features[position[0], position[1]]
        return self.colors_to_features[color]
    
    def _visualize_policy(self, policy, save_path="policy_visualization.png", title=None):
        """
        Visualize the given policy on the grid world, add a title, and save the image.
        
        Args:
            policy: A list of (state, action) tuples representing the policy to visualize.
            save_path: File path to save the visualization image.
            title: Optional title to display on the image.
        """
        self.render_mode = "rgb_array"  # Set the render mode to generate images
        pix_square_size = self.window_size / self.size  # Size of each grid square in pixels
        
        # Set up the PyGame window and canvas
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size + 50))  # Extra space for title
        canvas = pygame.Surface((self.window_size, self.window_size + 50))
        canvas.fill((255, 255, 255))  # White background

        # Draw the grid and agent's movements according to the policy
        self._draw_grid(canvas, pix_square_size)
        self._draw_gridlines(canvas, pix_square_size)

        # Action mappings to arrow symbols or directions
        action_arrows = {0: "^", 1: "v", 2: "<", 3: ">"}
        font = pygame.font.SysFont(None, 36)  # Font for arrows

        # Draw each state in the policy with its corresponding action
        for state, action in policy:
            if action is not None:  # Ensure the action is valid
                row, col = divmod(state, self.size)
                action_symbol = action_arrows.get(action, "?")
                # Render the action symbol (arrow) in the center of the cell
                text_surface = font.render(action_symbol, True, (0, 0, 0))  # Black arrows
                canvas.blit(text_surface, (col * pix_square_size + pix_square_size / 4, row * pix_square_size + pix_square_size / 4))

        # Add the title to the image (optional)
        if title:
            title_font = pygame.font.SysFont(None, 36)  # Choose font and size
            title_surface = title_font.render(title, True, (0, 0, 0))  # Render title in black
            canvas.blit(title_surface, (self.window_size // 2 - title_surface.get_width() // 2, self.window_size))  # Center the title below the grid

        # Save the final rendered image to the file
        pygame.image.save(canvas, save_path)
        print(f"Policy visualization saved to {save_path}")

        # Close the PyGame window
        pygame.display.quit()
