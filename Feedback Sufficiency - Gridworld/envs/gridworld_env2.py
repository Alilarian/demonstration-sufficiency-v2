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

    def __init__(self, gamma, color_to_feature_map, grid_features, render_mode=None, noise_prob=0.1, terminal_states=None, custom_feature_weights=None):
        """
        Initializes the GridWorld environment with user-defined features, grid layout, custom feature weights, and visualization settings.
        
        Args:
            gamma (float): Discount factor for MDP.
            color_to_feature_map (dict): Dictionary mapping color names to feature vectors.
            grid_features (list of lists): 2D list representing the grid layout with color names.
            render_mode (str, optional): Rendering mode. Defaults to None.
            noise_prob (float, optional): Noise probability for actions. Defaults to 0.1.
            terminal_states (list, optional): List of terminal state indices. Defaults to None.
            custom_feature_weights (list, optional): Custom weights for feature vectors. Defaults to None (random initialization).
        """
        super(NoisyLinearRewardFeaturizedGridWorldEnv, self).__init__()
        self.rows = len(grid_features)
        self.columns = len(grid_features[0])
        self.window_size = 512  # Window size for rendering
        self.noise_prob = noise_prob
        self.gamma = gamma

        # Pixel size for rendering
        self.pix_square_width = self.window_size / self.columns
        self.pix_square_height = self.window_size / self.rows

        # Set the feature map based on input
        self.colors_to_features = {color: np.array(features) for color, features in color_to_feature_map.items()}

        # Validate that all grid entries have corresponding feature vectors
        for row in grid_features:
            for color in row:
                assert color in self.colors_to_features, f"Color '{color}' in grid_features not defined in color_to_feature_map."

        self.grid_features = np.array(grid_features)

        #self.set_random_seed(42)
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, max(self.rows, self.columns) - 1, shape=(2,), dtype=int),
            "target": spaces.Box(0, max(self.rows, self.columns) - 1, shape=(2,), dtype=int),
        })
        self.action_space = spaces.Discrete(4)
        self.num_states = self.get_num_states()
        self.num_actions = self.get_num_actions()
        self.terminal_states = terminal_states if terminal_states else [self.num_states - 1]
        self.num_feat = len(next(iter(self.colors_to_features.values())))

        # Set custom feature weights if provided, otherwise use random initialization
        if custom_feature_weights:
            assert len(custom_feature_weights) == self.num_feat, "Custom feature weights must match the number of features."
            self.feature_weights = np.array(custom_feature_weights)
        else:
            self.feature_weights = sorted(np.random.randn(self.num_feat))

        self.transitions = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.initialize_transition_matrix()
        self._set_terminal_state_transitions()

        # Initialize rendering settings
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        assert render_mode is None or render_mode in self.metadata["render_modes"]


    def get_cell_features(self, position):
        """
        Returns the feature vector of the grid cell at the given position.
        """
        color = self.grid_features[position[0], position[1]]
        return self.colors_to_features[color]

    def _initialize_grid_features(self):
        # This method is not needed as we are using custom grid features
        pass

    # Other methods remain unchanged

    def initialize_transition_matrix(self):
            """
            Initializes the transition matrix with noisy state transitions.
            """
            RIGHT = 3
            UP = 0
            LEFT = 2
            DOWN = 1
            num_states = self.rows * self.columns

            for s in range(num_states):
                row, col = divmod(s, self.columns)

                # Transitions for UP
                if row > 0:
                    self.transitions[s][UP][s - self.columns] = 1.0 - (2 * self.noise_prob)
                else:
                    self.transitions[s][UP][s] = 1.0 - (2 * self.noise_prob)
                if col > 0:
                    self.transitions[s][UP][s - 1] = self.noise_prob
                else:
                    self.transitions[s][UP][s] = self.noise_prob
                if col < self.columns - 1:
                    self.transitions[s][UP][s + 1] = self.noise_prob
                else:
                    self.transitions[s][UP][s] = self.noise_prob

                # Handle top-left and top-right corners for UP
                if s < self.columns and col == 0:  # Top-left corner
                    self.transitions[s][UP][s] = 1.0 - self.noise_prob
                elif s < self.columns and col == self.columns - 1:  # Top-right corner
                    self.transitions[s][UP][s] = 1.0 - self.noise_prob
                
                # Transitions for DOWN
                if row < self.rows - 1:
                    self.transitions[s][DOWN][s + self.columns] = 1.0 - (2 * self.noise_prob)
                else:
                    self.transitions[s][DOWN][s] = 1.0 - (2 * self.noise_prob)
                if col > 0:
                    self.transitions[s][DOWN][s - 1] = self.noise_prob
                else:
                    self.transitions[s][DOWN][s] = self.noise_prob
                if col < self.columns - 1:
                    self.transitions[s][DOWN][s + 1] = self.noise_prob
                else:
                    self.transitions[s][DOWN][s] = self.noise_prob

                # Handle bottom-left and bottom-right corners for DOWN
                if s >= (self.rows - 1) * self.columns and col == 0:  # Bottom-left corner
                    self.transitions[s][DOWN][s] = 1.0 - self.noise_prob
                elif s >= (self.rows - 1) * self.columns and col == self.columns - 1:  # Bottom-right corner
                    self.transitions[s][DOWN][s] = 1.0 - self.noise_prob

                # Transitions for LEFT
                if col > 0:
                    self.transitions[s][LEFT][s - 1] = 1.0 - (2 * self.noise_prob)
                else:
                    self.transitions[s][LEFT][s] = 1.0 - (2 * self.noise_prob)
                if row > 0:
                    self.transitions[s][LEFT][s - self.columns] = self.noise_prob
                else:
                    self.transitions[s][LEFT][s] = self.noise_prob
                if row < self.rows - 1:
                    self.transitions[s][LEFT][s + self.columns] = self.noise_prob
                else:
                    self.transitions[s][LEFT][s] = self.noise_prob

                # Handle top-left and bottom-left corners for LEFT
                if s < self.columns and col == 0:  # Top-left corner
                    self.transitions[s][LEFT][s] = 1.0 - self.noise_prob
                elif s >= (self.rows - 1) * self.columns and col == 0:  # Bottom-left corner
                    self.transitions[s][LEFT][s] = 1.0 - self.noise_prob

                # Transitions for RIGHT
                if col < self.columns - 1:
                    self.transitions[s][RIGHT][s + 1] = 1.0 - (2 * self.noise_prob)
                else:
                    self.transitions[s][RIGHT][s] = 1.0 - (2 * self.noise_prob)
                if row > 0:
                    self.transitions[s][RIGHT][s - self.columns] = self.noise_prob
                else:
                    self.transitions[s][RIGHT][s] = self.noise_prob
                if row < self.rows - 1:
                    self.transitions[s][RIGHT][s + self.columns] = self.noise_prob
                else:
                    self.transitions[s][RIGHT][s] = self.noise_prob

                # Handle top-right and bottom-right corners for RIGHT
                if s < self.columns and col == self.columns - 1:  # Top-right corner
                    self.transitions[s][RIGHT][s] = 1.0 - self.noise_prob
                elif s >= (self.rows - 1) * self.columns and col == self.columns - 1:  # Bottom-right corner
                    self.transitions[s][RIGHT][s] = 1.0 - self.noise_prob

    def _set_terminal_state_transitions(self):
        for terminal_state in self.terminal_states:
            self.transitions[terminal_state, :, :] = 0
            self.transitions[terminal_state, :, terminal_state] = 1

    def get_num_states(self):
        return self.rows * self.columns
    
    def step(self, action):
        row, col = self._agent_location
        raw_index = row * self.columns + col

        # Sample the next state based on transition probabilities
        next_state = random.choices(range(self.rows * self.columns), self.transitions[raw_index][action])[0]
        new_row, new_col = divmod(next_state, self.columns)
        new_index = new_row * self.columns + new_col

        # Check if the agent stayed in the same cell
        if next_state == raw_index:
            reward = 0  # Assign a reward of 0 if the agent stays in the same state
        else:
            reward = self.compute_reward(new_index)

        # Update the agent's position
        self._agent_location = np.array([new_row, new_col])

        # Check if we reached a terminal state
        terminated = next_state in self.terminal_states

        observation = self.get_observation()

        if self.render_mode == "human":
            self.render_grid_frame()

        return observation, reward, terminated, False

    def reset(self, seed=None, options=None):
        #self.set_random_seed(seed)
        super().reset(seed=seed)

        while True:
            self._agent_location = self.np_random.integers(0, self.rows - 1, size=2)
            raw_index = self._agent_location[0] * self.columns + self._agent_location[1]
            if raw_index not in self.terminal_states:
                break

        observation = self.get_observation()

        if self.render_mode == "human":
            self.render_grid_frame()

        return observation

    def get_observation(self):
        return {"agent": self._agent_location, "terminal states": self.terminal_states}

    def compute_reward(self, state):
        row, col = divmod(state, self.columns)
        cell_features = self.get_cell_features([row, col])
        return np.dot(cell_features, self.feature_weights)

    def get_cell_features(self, position):
        color = self.grid_features[position[0], position[1]]
        return self.colors_to_features[color]

    def render_grid_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            # Set the window size dynamically based on grid dimensions
            self.window = pygame.display.set_mode((self.columns * self.pix_square_width, self.rows * self.pix_square_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Create a canvas with dimensions based on the number of rows and columns
        canvas = pygame.Surface((self.columns * self.pix_square_width, self.rows * self.pix_square_height))
        canvas.fill((255, 255, 255))  # White background

        self._draw_grid(canvas)
        self._draw_agent(canvas)
        self._draw_gridlines(canvas)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def _draw_grid(self, canvas):
        color_map = {
            "blue": (0, 0, 255),       # Blue
            "red": (255, 0, 0),        # Red
            "green": (0, 255, 0),      # Green
            "yellow": (255, 255, 0),   # Yellow
            "purple": (128, 0, 128),   # Purple
            "orange": (255, 165, 0),   # Orange

        }
        for terminal_state in self.terminal_states:
            row, col = divmod(terminal_state, self.columns)
            pygame.draw.rect(canvas, (0, 0, 0), pygame.Rect(self.pix_square_width * col, self.pix_square_height * row, self.pix_square_width, self.pix_square_height))

        for x in range(self.rows):
            for y in range(self.columns):
                if any([x == t_row and y == t_col for t_row, t_col in [divmod(ts, self.columns) for ts in self.terminal_states]]):
                    continue
                color = color_map[self.grid_features[x, y]]
                pygame.draw.rect(canvas, color, pygame.Rect(self.pix_square_width * y, self.pix_square_height * x, self.pix_square_width, self.pix_square_height))

    def _draw_agent(self, canvas):
        pygame.draw.circle(canvas, (42, 42, 42), ((self._agent_location[1] + 0.5) * self.pix_square_width, (self._agent_location[0] + 0.5) * self.pix_square_height), min(self.pix_square_width, self.pix_square_height) / 3)

    def _draw_gridlines(self, canvas):
        for x in range(self.rows + 1):
            pygame.draw.line(canvas, (0, 0, 0), (0, self.pix_square_height * x), (self.columns * self.pix_square_width, self.pix_square_height * x), width=3)
        for y in range(self.columns + 1):
            pygame.draw.line(canvas, (0, 0, 0), (self.pix_square_width * y, 0), (self.pix_square_width * y, self.rows * self.pix_square_height), width=3)

    def set_random_seed(self, seed):
        """Sets the random seed for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)

    def get_discount_factor(self):
        return self.gamma
    
    def get_num_actions(self):
        return self.action_space.n
    
    def get_num_states(self):
        return self.rows * self.columns

    def get_feature_weights(self):
        return self.feature_weights
    
    def set_feature_weights(self, weights):
        """Set and normalize a new weight vector for feature-based rewards."""
        self.feature_weights = weights / np.linalg.norm(weights)
        
    def compute_reward(self, state):
        """
        Computes the reward for a given state based on its feature vector and the feature weights.
        """
        row, col = divmod(state, self.columns)
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
        pix_square_size = self.window_size / max(self.rows, self.columns)

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
                row, col = divmod(state, self.columns)
                action_symbol = action_arrows.get(action, "?")
                # Render the action symbol (arrow) in the center of the cell
                text_surface = font.render(action_symbol, True, (0, 0, 0))  # Black arrows
                canvas.blit(text_surface, (col * pix_square_size + pix_square_size / 4, row * pix_square_size + pix_square_size / 4))

        # Add the title to the image (optional)
        if title:
            title_font = pygame.font.SysFont(None, 36)  # Choose font and size
            title_surface = title_font.render(title, True, (0, 0, 0))  # Render title in terminal
            canvas.blit(title_surface, (self.window_size // 2 - title_surface.get_width() // 2, self.window_size))  # Center the title below the grid

        # Save the final rendered image to the file
        pygame.image.save(canvas, save_path)
        print(f"Policy visualization saved to {save_path}")

        # Close the PyGame window
        pygame.display.quit()