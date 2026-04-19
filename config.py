"""
Configuration settings for Snake AI project.
"""

# Training configuration
TRAINING_CONFIG = {
    'num_episodes': 500,
    'max_memory': 100_000,
    'batch_size': 1000,
    'learning_rate': 0.001,
    'gamma': 0.9,
    'save_model': True,
    'plot_interval': 10
}

# Game configuration
GAME_CONFIG = {
    'width': 640,
    'height': 480,
    'block_size': 20,
    'speed': 20,
    'font_size': 25
}

# AI configuration
AI_CONFIG = {
    'state_size': 11,
    'hidden_size': 256,
    'action_size': 3,
    'epsilon_start': 80,
    'epsilon_decay': True
}

# File paths
PATHS = {
    'model_dir': 'models',
    'model_file': 'models/model.pth',
    'font_file': 'assets/fonts/arial.ttf',
    'training_plot': 'training_results.png'
}

# Visualization settings
VIZ_CONFIG = {
    'figure_size': (12, 6),
    'dpi': 300,
    'grid_alpha': 0.3
}
