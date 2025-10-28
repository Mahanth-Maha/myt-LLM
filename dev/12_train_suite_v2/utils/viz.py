import os
import json
import matplotlib.pyplot as plt
import numpy as np


def create_checkpoint_plots(metrics_history, output_dir, step):
    os.makedirs(output_dir, exist_ok=True)

    if not metrics_history or len(metrics_history.get('steps', [])) < 2:
        return

    steps = metrics_history['steps']

    save_individual_plots(metrics_history, output_dir, step)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f'Training Metrics - Step {step}', fontsize=16, fontweight='bold')

    if 'train_loss' in metrics_history and 'val_loss' in metrics_history:
        axes[0, 0].plot(steps, metrics_history['train_loss'],
                        label='Train Loss', color='blue', linewidth=2)
        axes[0, 0].plot(steps, metrics_history['val_loss'],
                        label='Val Loss', color='red', linewidth=2)
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training vs Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    if 'perplexity' in metrics_history:
        log_scaled_prep = []
        for val in metrics_history['perplexity']:
            if val <= 0:
                log_scaled_prep.append(1e-8)
            else:
                log_scaled_prep.append(val)

        axes[0, 1].plot(steps, log_scaled_prep,
                        label='Perplexity', color='green', linewidth=2)
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Perplexity')
        axes[0, 1].set_title('Perplexity Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
    if 'accuracy' in metrics_history:
        axes[0, 2].plot(steps, metrics_history['accuracy'],
                        label='Accuracy', color='purple', linewidth=2)
        axes[0, 2].set_xlabel('Steps')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].set_title('Validation Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim(0, 1)
    if 'tokens_per_second' in metrics_history:
        axes[1, 0].plot(steps, metrics_history['tokens_per_second'],
                        label='Tokens/sec', color='orange', linewidth=2)
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Tokens per Second')
        axes[1, 0].set_title('Training Throughput')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    if 'learning_rate' in metrics_history:
        axes[1, 1].plot(steps, metrics_history['learning_rate'],
                        label='Learning Rate', color='brown', linewidth=2)
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
    if 'train_loss' in metrics_history:
        train_loss = np.array(metrics_history['train_loss'])
        window = min(50, len(train_loss) // 10)
        if window > 1:
            smoothed_loss = np.convolve(
                train_loss, np.ones(window)/window, mode='valid')
            smoothed_steps = steps[window-1:]
            axes[1, 2].plot(smoothed_steps, smoothed_loss,
                            label=f'Smoothed Loss (window={window})', color='teal', linewidth=2)
            axes[1, 2].set_xlabel('Steps')
            axes[1, 2].set_ylabel('Smoothed Loss')
            axes[1, 2].set_title('Loss Trend (Moving Average)')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'training_metrics_step_{step}.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def save_individual_plots(metrics_history, output_dir, step):

    steps = metrics_history['steps']

    if 'train_loss' in metrics_history and 'val_loss' in metrics_history:
        plt.figure(figsize=(12, 8))
        plt.plot(steps, metrics_history['train_loss'],
                 label='Train Loss', color='blue', linewidth=2)
        plt.plot(steps, metrics_history['val_loss'],
                 label='Val Loss', color='red', linewidth=2)
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Cross-Entropy Loss', fontsize=12)
        plt.title(
            f'Training and Validation Loss - Step {step}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'loss_curves_step_{step}.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    if 'perplexity' in metrics_history:
        plt.figure(figsize=(12, 8))
        plt.plot(steps, metrics_history['perplexity'],
                 label='Perplexity', color='green', linewidth=2)
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Perplexity', fontsize=12)
        plt.title(
            f'Perplexity Over Time - Step {step}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        # plt.yscale('log')
        plt.savefig(os.path.join(output_dir, f'perplexity_step_{step}.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    if 'tokens_per_second' in metrics_history:
        plt.figure(figsize=(12, 8))
        plt.plot(steps, metrics_history['tokens_per_second'],
                 label='Throughput', color='orange', linewidth=2)
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Tokens per Second', fontsize=12)
        plt.title(
            f'Training Throughput - Step {step}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'throughput_step_{step}.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


def plot_metrics(metric_logs, output_dir = "./checkpoints/milestones/plots"):
    os.makedirs(output_dir, exist_ok=True)

    with open(metric_logs, 'r') as f:
        history = json.load(f)

    step = history.get('step', 0)
    metrics_history = history.get('metrics_history', history)

    create_checkpoint_plots(metrics_history, output_dir, step)
