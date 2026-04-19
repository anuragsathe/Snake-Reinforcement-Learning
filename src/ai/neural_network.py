"""
Deep Q-Network implementation for Snake game AI.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from pathlib import Path
from typing import Union


class DeepQNetwork(nn.Module):
    """Deep Q-Network for learning Snake game strategy."""
    
    def __init__(self, input_size: int = 11, hidden_size: int = 256, output_size: int = 3):
        """Initialize the neural network.
        
        Args:
            input_size: Number of input features (game state)
            hidden_size: Number of neurons in hidden layer
            output_size: Number of possible actions
        """
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output Q-values
        """
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name: str = 'model.pth') -> None:
        """Save the model weights to disk.
        
        Args:
            file_name: Name of the model file
        """
        model_dir = Path('models')
        model_dir.mkdir(exist_ok=True)
        
        file_path = model_dir / file_name
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load(self, file_name: str = 'model.pth') -> bool:
        """Load model weights from disk.
        
        Args:
            file_name: Name of the model file
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        # Handle both relative and absolute paths
        if os.path.isabs(file_name):
            file_path = Path(file_name)
        else:
            model_dir = Path('models')
            file_path = model_dir / file_name
        
        if file_path.exists():
            self.load_state_dict(torch.load(str(file_path)))
            print(f"Model loaded from {file_path}")
            return True
        else:
            print(f"No model found at {file_path}")
            return False


class QTrainer:
    """Q-Learning trainer for the Deep Q-Network."""
    
    def __init__(self, model, learning_rate=0.001, gamma=0.9):
        """Initialize the trainer.
        
        Args:
            model: The neural network model to train
            learning_rate: Learning rate for optimization
            gamma: Discount factor for future rewards
        """
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """Perform one training step.
        
        Args:
            state: Current game state
            action: Action taken
            reward: Reward received
            next_state: Next game state
            done: Whether the episode is finished
        """
        # Convert to tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Handle single state vs batch
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: Predict Q values with current state
        prediction = self.model(state)

        # 2: Calculate target Q values
        target = prediction.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 3: Calculate loss and backpropagate
        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()
        self.optimizer.step()
