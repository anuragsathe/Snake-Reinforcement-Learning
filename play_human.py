"""
Script to play Snake game with human controls.
"""

import pygame
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.game import SnakeGame


def main():
    """Main function for human play mode."""
    print("=== Snake Game - Human Mode ===")
    print("Controls:")
    print("  Arrow Keys: Move snake")
    print("  Close window: Quit game")
    print("\nStarting game...")
    
    try:
        game = SnakeGame()
        
        while True:
            reward, done, score = game.play_step()
            
            if done:
                print(f"\nGame Over! Final Score: {score}")
                print("Press any key to play again or close window to quit...")
                
                # Wait for input or window close
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return
                        elif event.type == pygame.KEYDOWN:
                            game.reset()
                            waiting = False
                            break
        
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pygame.quit()


if __name__ == '__main__':
    main()
