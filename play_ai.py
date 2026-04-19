"""
Script to watch the trained AI play Snake game.
"""

import pygame
import sys
import os
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ai import DQNAgent
from src.game import SnakeGame


def watch_ai_play(model_path='model.pth', num_games=5, delay=0.1):
    """Watch the trained AI play Snake game.
    
    Args:
        model_path: Path to the trained model file
        num_games: Number of games to watch
        delay: Delay between moves for better visualization
    """
    print("=== Snake AI - Watch Mode ===")
    print(f"Loading model from: {model_path}")
    
    # Initialize pygame
    pygame.init()
    
    try:
        # Initialize agent and game
        agent = DQNAgent()
        
        # Load trained model
        if not agent.load_model(model_path):
            print("No trained model found. Please train the agent first using train.py")
            return
        
        game = SnakeGame()
        
        # Set game speed for better visualization
        import src.game.constants as constants
        constants.SPEED = max(1, int(constants.SPEED * delay))
        
        total_score = 0
        best_score = 0
        
        for game_num in range(num_games):
            print(f"\nGame {game_num + 1}/{num_games}")
            game.reset()
            
            while True:
                # Get current state
                state = agent.get_state(game)
                
                # Get AI action (no exploration)
                agent.epsilon = 0  # Disable exploration
                action = agent.get_action(state)
                
                # Perform action
                reward, done, score = game.play_step(action)
                
                if done:
                    total_score += score
                    best_score = max(best_score, score)
                    print(f"Score: {score}")
                    break
                
                # Small delay for better visualization
                time.sleep(delay)
        
        print(f"\n=== Results ===")
        print(f"Games played: {num_games}")
        print(f"Total score: {total_score}")
        print(f"Average score: {total_score / num_games:.2f}")
        print(f"Best score: {best_score}")
        
    except KeyboardInterrupt:
        print("\nWatching interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pygame.quit()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Watch trained AI play Snake')
    parser.add_argument('--model', '-m', default='model.pth',
                       help='Path to trained model file')
    parser.add_argument('--games', '-g', type=int, default=5,
                       help='Number of games to watch')
    parser.add_argument('--delay', '-d', type=float, default=0.1,
                       help='Delay between moves (seconds)')
    
    args = parser.parse_args()
    
    watch_ai_play(args.model, args.games, args.delay)


if __name__ == '__main__':
    main()
