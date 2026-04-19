"""
Visualization utilities for training progress and game statistics.
"""

import matplotlib.pyplot as plt
from IPython import display

plt.ion()


def plot_training_progress(scores, mean_scores, title="Snake AI Training Progress"):
    """Plot training progress with scores and mean scores.
    
    Args:
        scores: List of individual game scores
        mean_scores: List of running mean scores
        title: Plot title
    """
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title(title)
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score', alpha=0.7)
    plt.plot(mean_scores, label='Mean Score', linewidth=2)
    plt.ylim(ymin=0)
    plt.legend()
    
    # Add latest score annotations
    if scores:
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    
    plt.grid(True, alpha=0.3)
    plt.show(block=False)
    plt.pause(.1)


def plot_final_results(scores, mean_scores, save_path=None):
    """Create a final plot of training results.
    
    Args:
        scores: List of individual game scores
        mean_scores: List of running mean scores
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(scores, label='Score', alpha=0.7, color='blue')
    plt.plot(mean_scores, label='Mean Score', linewidth=2, color='red')
    plt.title('Snake AI Training Results')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
