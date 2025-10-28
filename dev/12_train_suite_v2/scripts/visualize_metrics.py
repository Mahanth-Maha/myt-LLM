import argparse
from box import Box
from utils.viz import plot_metrics, create_checkpoint_plots

import os
from dotenv import load_dotenv
load_dotenv() 


def main():
    parser = argparse.ArgumentParser(description="Visualize MYT-LLM Metrics")
    parser.add_argument("--metrics", type=str, required=True, help="Path to metrics.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/milestones/plots", help="Output folder")
    args = parser.parse_args()

    cfg = Box(vars(args))

    plot_metrics(cfg.metrics, output_dir=cfg.output_dir)
    
    with open(args.metrics) as f:
        data = __import__('json').load(f)
    history = data.get('metrics_history', None)
    step = data.get('step', 0)
    if history:
        create_checkpoint_plots(history, args.output_dir, step)
    
    print(f"Plots saved to {cfg.output_dir}")

if __name__ == "__main__":
    main()
