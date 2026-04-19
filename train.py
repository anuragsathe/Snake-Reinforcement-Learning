"""
Main training script for the Snake AI agent.
"""

import pygame
import sys
import os
from typing import Tuple, List

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ai import DQNAgent
from src.game import SnakeGame
from src.utils import plot_training_progress


def train_agent(
    num_episodes: int = 1000,
    max_memory: int = 100_000,
    batch_size: int = 1000,
    learning_rate: float = 0.001,
    gamma: float = 0.9,
    save_model: bool = True,
    plot_interval: int = 10
) -> Tuple[DQNAgent, List[int], List[float]]:
    """Train the DQN agent to play Snake.
    
    Args:
        num_episodes: Number of training episodes
        max_memory: Maximum replay memory size
        batch_size: Training batch size
        learning_rate: Neural network learning rate
        gamma: Discount factor for future rewards
        save_model: Whether to save the best model
        plot_interval: Update plot every N games
    
    Returns:
        Tuple[DQNAgent, List[int], List[float]]: Trained agent and performance metrics
    """
    # Validate inputs
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if not (0 < learning_rate < 1):
        raise ValueError("learning_rate must be between 0 and 1")
    if not (0 < gamma < 1):
        raise ValueError("gamma must be between 0 and 1")
    
    try:
        # Initialize pygame
        pygame.init()
        
        # Initialize agent and game
        agent = DQNAgent(
            max_memory=max_memory,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gamma=gamma
        )
        game = SnakeGame()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize training components: {e}")
    
    # Training metrics
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record_score = 0
    
    print(f"Starting training for {num_episodes} episodes...")
    print(f"Agent parameters: LR={learning_rate}, Gamma={gamma}, Batch={batch_size}")
    
    for episode in range(num_episodes):
        # Get current state
        state_old = agent.get_state(game)
        
        # Choose action
        action = agent.get_action(state_old)
        
        # Perform action and get feedback
        reward, done, score = game.play_step(action)
        state_new = agent.get_state(game)
        
        # Train short memory (single step)
        agent.train_short_memory(state_old, action, reward, state_new, done)
        
        # Store experience
        agent.remember(state_old, action, reward, state_new, done)
        
        if done:
            # Episode finished
            game.reset()
            agent.n_games += 1
            
            # Train long memory (experience replay)
            agent.train_long_memory()
            
            # Update record
            if score > record_score:
                record_score = score
                if save_model:
                    agent.save_model()
            
            # Update metrics
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            # Print progress
            if episode % plot_interval == 0 or episode == num_episodes - 1:
                print(f"Episode {agent.n_games:4d} | Score: {score:3d} | "
                      f"Record: {record_score:3d} | Mean: {mean_score:.2f} | "
                      f"Epsilon: {agent.epsilon:.2f}")
                
                # Update plot
                plot_training_progress(plot_scores, plot_mean_scores)
    
    print(f"\nTraining completed!")
    print(f"Final record score: {record_score}")
    print(f"Final mean score: {total_score / agent.n_games:.2f}")
    print(f"Total episodes played: {agent.n_games}")
    
    pygame.quit()
    return agent, plot_scores, plot_mean_scores


def main():
    """Main function with configuration."""
    # Training configuration
    config = {
        'num_episodes': 500,
        'max_memory': 100_000,
        'batch_size': 1000,
        'learning_rate': 0.001,
        'gamma': 0.9,
        'save_model': True,
        'plot_interval': 10
    }
    
    # Start training
    try:
        agent, scores, mean_scores = train_agent(**config)
        
        # Save final results
        from src.utils import plot_final_results
        plot_final_results(scores, mean_scores, save_path='training_results.png')
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        pygame.quit()
    except Exception as e:
        print(f"Error during training: {e}")
        pygame.quit()


if __name__ == '__main__':
    main()
