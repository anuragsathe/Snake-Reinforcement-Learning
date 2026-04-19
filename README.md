# Snake-Reinforcement-Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)
![Pygame](https://img.shields.io/badge/Pygame-2.0%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

*Master the classic Snake game with Deep Q-Learning AI*

[![Demo](https://img.shields.io/badge/Demo-View-red)](#demo)
[![Documentation](https://img.shields.io/badge/Docs-Read-blue)](#documentation)
[![Installation](https://img.shields.io/badge/Install-Quick-green)](#installation)

</div>

## Overview

This project implements a sophisticated Deep Q-Network (DQN) that learns to play Snake from scratch using reinforcement learning. Watch as the AI evolves from random movements to strategic gameplay through experience replay and neural network training.

## Features

- **Advanced AI Architecture**: Deep Q-Network with experience replay for stable learning
- **Intelligent State Representation**: 11-dimensional state space with danger detection and food tracking
- **Professional Codebase**: Modular design with comprehensive type hints and documentation
- **Multiple Game Modes**: Human play, AI training, and AI demonstration
- **Real-time Visualization**: Live training progress plots and performance metrics
- **Model Persistence**: Save and load trained models for continued learning
- **Extensible Design**: Easy to modify and extend with new algorithms

## Quick Start

```bash
# Clone and setup
git clone https://github.com/anuragsathe/Snake-Reinforcement-Learning.git
cd Snake-Reinforcement-Learning
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Train the AI
python train.py

# Watch it play
python play_ai.py --games 10
```

## Demo

### Training Progress
The AI learns from random movements to strategic gameplay:

```
Episode    50 | Score:   5 | Record:   8 | Mean: 3.12 | Epsilon: 30.00
Episode   100 | Score:  12 | Record:  15 | Mean: 7.45 | Epsilon: 20.00
Episode   200 | Score:  18 | Record:  22 | Mean: 11.23 | Epsilon: 0.00
Episode   300 | Score:  25 | Record:  28 | Mean: 15.67 | Epsilon: 0.00
```

### AI Performance
Watch the trained agent demonstrate advanced strategies:
- **Obstacle Avoidance**: Intelligent navigation around walls and self
- **Food Seeking**: Efficient pathfinding to food items
- **Survival Tactics**: Long-term survival strategies

## Project Structure

```
Snake-Reinforcement-Learning/
|
+-- src/                       # Source code
|   +-- ai/                   # AI components
|   |   +-- dqn_agent.py      # Deep Q-Learning agent
|   |   +-- neural_network.py # DQN architecture & trainer
|   |   +-- __init__.py
|   |
|   +-- game/                 # Game engine
|   |   +-- snake_game.py     # Snake game implementation
|   |   +-- constants.py      # Game configuration
|   |   +-- __init__.py
|   |
|   +-- utils/                # Utilities
|   |   +-- visualization.py  # Plotting & visualization
|   |   +-- __init__.py
|   |
|   +-- core/                 # Core components
|       +-- __init__.py
|
+-- assets/fonts/             # Font files
+-- models/                   # Saved models (auto-created)
+-- tests/                    # Test files
+-- docs/                     # Documentation
|
+-- train.py                  # Main training script
+-- play_human.py             # Human play mode
+-- play_ai.py                # AI demonstration
+-- config.py                 # Configuration settings
+-- requirements.txt          # Dependencies
+-- README.md                 # This file
```

## Installation

### Prerequisites
- Python 3.7 or higher
- 4GB RAM minimum (8GB recommended)
- Optional: GPU for faster training

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/anuragsathe/Snake-Reinforcement-Learning.git
   cd Snake-Reinforcement-Learning
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Unix/MacOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the AI

```bash
python train.py
```

**Training Features:**
- 500 episodes by default (configurable)
- Real-time progress visualization
- Automatic model saving on new records
- Experience replay for stable learning

**Custom Training:**
```bash
python train.py --episodes 1000 --lr 0.0005 --gamma 0.95
```

### Human Play Mode

```bash
python play_human.py
```

**Controls:**
- Arrow Keys: Move snake
- Close window: Quit

### AI Demonstration

```bash
python play_ai.py --games 10 --delay 0.05
```

**Options:**
- `--model`: Path to trained model
- `--games`: Number of games to watch
- `--delay`: Delay between moves

## AI Architecture

### State Representation (11 Features)

The AI perceives the game through 11 binary features:

1. **Danger Detection** (3 features)
   - Danger straight ahead
   - Danger to the right  
   - Danger to the left

2. **Current Direction** (4 features)
   - Moving left, right, up, or down

3. **Food Location** (4 features)
   - Food position relative to snake head

### Neural Network Architecture

```
Input Layer (11 neurons)
        |
        v
Hidden Layer (256 neurons, ReLU)
        |
        v
Output Layer (3 neurons, Q-values)
```

### Action Space

Three possible actions (one-hot encoded):
- `[1, 0, 0]`: Turn left
- `[0, 1, 0]`: Go straight  
- `[0, 0, 1]`: Turn right

### Training Algorithm

**Deep Q-Learning with:**
- **Experience Replay**: Stores and samples past experiences
- **Epsilon-Greedy Exploration**: Balances exploration vs exploitation
- **Target Q-Values**: Bellman equation for temporal difference learning
- **Adam Optimizer**: Efficient gradient-based optimization

## Configuration

### Training Parameters

```python
TRAINING_CONFIG = {
    'num_episodes': 500,        # Training episodes
    'max_memory': 100_000,      # Replay memory size
    'batch_size': 1000,         # Training batch size
    'learning_rate': 0.001,     # Neural network learning rate
    'gamma': 0.9,               # Discount factor
    'save_model': True,         # Save best models
    'plot_interval': 10         # Update plot frequency
}
```

### Game Settings

```python
GAME_CONFIG = {
    'width': 640,               # Window width
    'height': 480,              # Window height
    'block_size': 20,           # Snake segment size
    'speed': 20,                # Game speed (FPS)
    'font_size': 25             # Score display size
}
```

## Performance

### Learning Progression

| Training Phase | Score Range | Behavior |
|----------------|-------------|----------|
| **Initial** (0-50 episodes) | 0-5 | Random movements |
| **Basic Learning** (50-150) | 5-15 | Food seeking |
| **Intermediate** (150-300) | 15-30 | Obstacle avoidance |
| **Advanced** (300+) | 30+ | Strategic gameplay |

### Performance Factors

- **Training Duration**: More episodes = better performance
- **Hyperparameters**: Learning rate, gamma, batch size
- **Exploration Strategy**: Epsilon decay schedule
- **Random Seed**: Initial conditions affect learning

## Advanced Usage

### Custom Agent Configuration

```python
from src.ai import DQNAgent

agent = DQNAgent(
    max_memory=200_000,
    batch_size=2000,
    learning_rate=0.0005,
    gamma=0.95
)
```

### Model Analysis

```python
# Load and analyze trained model
agent = DQNAgent()
agent.load_model('models/model.pth')

# Inspect state representation
game = SnakeGame()
state = agent.get_state(game)
print(f"State shape: {state.shape}")
print(f"State features: {state}")
```

### Custom Visualization

```python
from src.utils import plot_final_results

# Generate performance plots
plot_final_results(scores, mean_scores, save_path='performance.png')
```

## Technical Details

### Dependencies

- **PyTorch** (1.9+): Deep learning framework
- **Pygame** (2.0+): Game engine and rendering
- **Matplotlib** (3.3+): Training visualization
- **NumPy** (1.20+): Numerical computations
- **IPython** (7.0+): Interactive plotting

### System Requirements

- **Python**: 3.7 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **GPU**: Optional but recommended for faster training
- **Display**: 640x480 minimum resolution

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Font Loading Issues**
   - Ensure `arial.ttf` is in `assets/fonts/`
   - Game falls back to system font if needed

3. **Slow Training**
   - Reduce `batch_size` in configuration
   - Decrease game speed in constants
   - Enable GPU acceleration

4. **Model Saving Issues**
   - Ensure `models/` directory exists
   - Check file permissions

### Performance Optimization

- **GPU Training**: PyTorch automatically uses CUDA if available
- **Memory Management**: Adjust `max_memory` based on available RAM
- **Visualization**: Increase `plot_interval` to reduce overhead

## Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Implement** your changes with proper documentation
4. **Add** tests if applicable
5. **Submit** a pull request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to new functions
- Include comprehensive docstrings
- Test your changes thoroughly

## License

This project is licensed under the MIT License.

## Acknowledgments

- **Deep Q-Learning**: Based on Mnih et al. (2015) "Human-level control through deep reinforcement learning"
- **Snake Game**: Inspired by the classic arcade game
- **Pygame Community**: For the excellent game development framework
- **PyTorch Team**: For the powerful deep learning framework

## Future Enhancements

- **Advanced Architectures**: Double DQN, Dueling DQN, Prioritized Experience Replay
- **Multi-Agent Training**: Competitive and cooperative scenarios
- **Web Interface**: Browser-based game and training visualization
- **Hyperparameter Optimization**: Automated tuning with Optuna or Ray Tune
- **Performance Metrics**: Detailed analysis and benchmarking tools
- **Curriculum Learning**: Progressive difficulty scaling

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{snake-ai-dqn,
  title={Snake AI: Deep Q-Learning Implementation},
  author={Anurag Sathe},
  year={2024},
  url={https://github.com/anuragsathe/Snake-Reinforcement-Learning}
}
```

---

<div align="center">

**[![Star](https://img.shields.io/github/stars/anuragsathe/Snake-Reinforcement-Learning.svg?style=social&label=Star)](https://github.com/anuragsathe/Snake-Reinforcement-Learning)**
**[![Fork](https://img.shields.io/github/forks/anuragsathe/Snake-Reinforcement-Learning.svg?style=social&label=Fork)](https://github.com/anuragsathe/Snake-Reinforcement-Learning/fork)**
**[![Watch](https://img.shields.io/github/watchers/anuragsathe/Snake-Reinforcement-Learning.svg?style=social&label=Watch)](https://github.com/anuragsathe/Snake-Reinforcement-Learning)**

*Happy Training! Watch your AI evolve from random movements to strategic gameplay.* 

</div>