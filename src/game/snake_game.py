"""
Snake game engine with support for both human and AI players.
"""

import pygame
import random
from enum import Enum
from collections import namedtuple

from .constants import *


class Direction(Enum):
    """Enumeration for snake movement directions."""
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')


class SnakeGame:
    """Main Snake game class supporting both human and AI control."""
    
    def __init__(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
        """Initialize the Snake game.
        
        Args:
            width: Game window width in pixels
            height: Game window height in pixels
        """
        self.width = width
        self.height = height
        
        # Initialize display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake AI')
        self.clock = pygame.time.Clock()
        
        # Initialize font
        pygame.font.init()  # Ensure font module is initialized
        try:
            self.font = pygame.font.Font(FONT_PATH, FONT_SIZE)
        except:
            self.font = pygame.font.SysFont('arial', FONT_SIZE)
        
        # Initialize game state
        self.reset()
        
    def reset(self):
        """Reset the game to initial state."""
        self.direction = Direction.RIGHT
        self.head = Point(self.width // 2, self.height // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        self.score = 0
        self.food = None
        self._place_food()
        
    def _place_food(self):
        """Place food at a random position not occupied by the snake."""
        x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        
        # Ensure food doesn't spawn on snake
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action=None):
        """Execute one game step.
        
        Args:
            action: AI action as [left, straight, right] one-hot encoded.
                   If None, game accepts human input.
        
        Returns:
            tuple: (reward, done, score)
        """
        # Handle input
        self._handle_input(action)
        
        # Move snake
        self._move(self.direction)
        self.snake.insert(0, self.head)
        
        # Check game over conditions
        reward = 0
        done = False
        
        if self._is_collision():
            done = True
            reward = -10
            return reward, done, self.score
            
        # Check if food was eaten
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # Update UI
        self._update_ui()
        self.clock.tick(SPEED)
        
        return reward, done, self.score
    
    def _handle_input(self, action):
        """Handle either AI actions or human keyboard input."""
        if action is None:
            # Human control
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.direction = Direction.LEFT
                    elif event.key == pygame.K_RIGHT:
                        self.direction = Direction.RIGHT
                    elif event.key == pygame.K_UP:
                        self.direction = Direction.UP
                    elif event.key == pygame.K_DOWN:
                        self.direction = Direction.DOWN
        else:
            # AI control - action is [left, straight, right]
            if action[0] == 1:  # Turn left
                self.direction = self._turn_left(self.direction)
            elif action[2] == 1:  # Turn right
                self.direction = self._turn_right(self.direction)
            # action[1] == 1 means go straight (no direction change)
    
    def _turn_left(self, current_direction):
        """Get direction when turning left from current direction."""
        turns = {
            Direction.RIGHT: Direction.UP,
            Direction.LEFT: Direction.DOWN,
            Direction.UP: Direction.LEFT,
            Direction.DOWN: Direction.RIGHT
        }
        return turns[current_direction]
    
    def _turn_right(self, current_direction):
        """Get direction when turning right from current direction."""
        turns = {
            Direction.RIGHT: Direction.DOWN,
            Direction.LEFT: Direction.UP,
            Direction.UP: Direction.RIGHT,
            Direction.DOWN: Direction.LEFT
        }
        return turns[current_direction]
    
    def _is_collision(self):
        """Check if snake has collided with boundaries or itself."""
        # Check boundary collision
        if (self.head.x >= self.width - BLOCK_SIZE or 
            self.head.x < 0 or 
            self.head.y >= self.height - BLOCK_SIZE or 
            self.head.y < 0):
            return True
        
        # Check self collision
        if self.head in self.snake[1:]:
            return True
            
        return False
    
    def is_collision(self, point=None):
        """Check collision at a specific point or at snake head.
        
        Args:
            point: Point to check collision. If None, checks snake head.
        
        Returns:
            bool: True if collision detected
        """
        if point is None:
            return self._is_collision()
        
        # Check boundary collision
        if (point.x >= self.width - BLOCK_SIZE or 
            point.x < 0 or 
            point.y >= self.height - BLOCK_SIZE or 
            point.y < 0):
            return True
        
        # Check snake collision
        if point in self.snake:
            return True
            
        return False
    
    def _update_ui(self):
        """Update the game display."""
        self.display.fill(BLACK)
        
        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
            
        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Draw score
        text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        
        pygame.display.flip()
    
    def _move(self, direction):
        """Move the snake head in the specified direction."""
        x, y = self.head.x, self.head.y
        
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)


def main():
    """Main function for human play mode."""
    pygame.init()
    game = SnakeGame()
    
    while True:
        reward, done, score = game.play_step()
        
        if done:
            break
        
    print(f'Final Score: {score}')
    pygame.quit()


if __name__ == '__main__':
    main()
