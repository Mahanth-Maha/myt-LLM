import os
import json
import torch
from utils.logging import setup_logging
from utils.metrics import compute_loss_perplexity, tokens_per_second

class CheckpointCallback:
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = setup_logging(cfg, name='train')

    def __call__(self, model, optimizer, scheduler, progress, step):
        from utils.checkpointing import save_checkpoint
        save_checkpoint(model, optimizer, scheduler, progress, self.cfg, step)
        self.logger.info(f"Checkpoint saved at step {step}")


class MilestoneCallback:
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = setup_logging(cfg, name='train')
        self.metrics_history = {
            'steps': [],
            'loss': [],
            'perplexity': [],
            'tokens_per_second': [],
            'learning_rate': [],
            'timestamp': []
        }

    def update_metrics(self, step, loss, learning_rate, tps, timestamp):
        """Update metrics history"""
        self.metrics_history['steps'].append(step)
        self.metrics_history['loss'].append(loss)
        self.metrics_history['perplexity'].append(compute_loss_perplexity(loss))
        self.metrics_history['tokens_per_second'].append(tps)
        self.metrics_history['learning_rate'].append(learning_rate)
        self.metrics_history['timestamp'].append(timestamp)

    def __call__(self, model, optimizer, scheduler, progress, step, **kwargs):
        if step % self.cfg['milestones']['interval_steps'] == 0:
            ms_dir = os.path.join(self.cfg['milestones']['dir'], f"step_{step}")
            os.makedirs(ms_dir, exist_ok=True)
            
            
            torch.save(model.state_dict(), os.path.join(ms_dir, 'model.pt'))
            torch.save(optimizer.state_dict(), os.path.join(ms_dir, 'optimizer.pt'))
            torch.save(scheduler.state_dict(), os.path.join(ms_dir, 'scheduler.pt'))
            
            
            dist_dir = os.path.join(ms_dir, self.cfg['milestones']['dist_subfolder'])
            os.makedirs(dist_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(dist_dir, 'model_inference.pt'))
            
            
            metrics_data = {
                'step': step,
                'progress': progress,
                'metrics_history': self.metrics_history,
                'current_metrics': {
                    'loss': kwargs.get('loss', 0.0),
                    'perplexity': kwargs.get('perplexity', 0.0),
                    'tokens_per_second': kwargs.get('tokens_per_second', 0.0),
                    'learning_rate': kwargs.get('learning_rate', 0.0)
                }
            }
            
            with open(os.path.join(ms_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            self.logger.info(f"Milestone saved at step {step}")

class ResumeCallback:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, progress):
        
        latest_dir = self.cfg['checkpoint']['dir']
        with open(os.path.join(latest_dir, 'progress.json'), 'w') as f:
            json.dump(progress, f, indent=2)
