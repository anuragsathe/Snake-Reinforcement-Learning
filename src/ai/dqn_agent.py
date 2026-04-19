"""
Deep Q-Learning Agent for playing Snake game.
"""

import torch
import random
import numpy as np
from collections import deque
from typing import List, Tuple, Optional

from ..game import SnakeGame, Direction, Point
from .neural_network import DeepQNetwork, QTrainer


class DQNAgent:
    """Deep Q-Learning Agent for Snake game."""
    
    def __init__(self, max_memory=100_000, batch_size=1000, learning_rate=0.001, gamma=0.9):
        """Initialize the DQN Agent.
        
        Args:
            max_memory: Maximum size of experience replay memory
            batch_size: Batch size for training
            learning_rate: Learning rate for neural network
            gamma: Discount factor for future rewards
        """
        self.max_memory = max_memory
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Agent state
        self.n_games = 0
        self.epsilon = 0  # Randomness factor
        
        # Experience replay memory
        self.memory = deque(maxlen=max_memory)
        
        # Neural network and trainer
        self.model = DeepQNetwork(11, 256, 3)
        self.trainer = QTrainer(self.model, learning_rate=learning_rate, gamma=gamma)

    def get_state(self, game: SnakeGame) -> np.ndarray:
        """Extract game state representation.
        
        Args:
            game: SnakeGame instance
        
        Returns:
            np.ndarray: Binary state representation with shape (11,)
        """
        head = game.snake[0]
        
        # Points around the head
        from ..game.constants import BLOCK_SIZE
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        # Current direction
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # State features
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location relative to head
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y   # food down
        ]

        return np.array(state, dtype=int)
    
    def remember(self, state: np.ndarray, action: List[int], reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """Store experience in replay memory.
        
        Args:
            state: Current game state
            action: Action taken
            reward: Reward received
            next_state: Next game state
            done: Whether episode is finished
        """
        self.memory.append((state, action, reward, next_state, done))
        
    def train_long_memory(self):
        """Train the model on a batch of experiences."""
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    def train_short_memory(self, state, action, reward, next_state, done):
        """Train the model on a single experience."""
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state: np.ndarray) -> List[int]:
        """Choose action using epsilon-greedy policy.
        
        Args:
            state: Current game state
        
        Returns:
            List[int]: One-hot encoded action [left, straight, right]
        """
        # Epsilon-greedy exploration (starts high, decreases over time)
        self.epsilon = max(0, 80 - self.n_games)  # Ensure epsilon doesn't go negative
        final_move = [0, 0, 0]
        
        if random.randint(0, 200) < self.epsilon:
            # Random exploration
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Exploitation - use neural network prediction
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
    def load_model(self, file_name: str = 'model.pth') -> bool:
        """Load a trained model.
        
        Args:
            file_name: Name of the model file
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        return self.model.load(file_name)
    
    def save_model(self, file_name: str = 'model.pth') -> None:
        """Save the current model.
        
        Args:
            file_name: Name of the model file
        """
        self.model.save(file_name)
