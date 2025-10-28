import os
import json
import torch
import shutil
from utils.viz import create_checkpoint_plots


def save_checkpoint(model, optimizer, scheduler, progress, cfg, step, samples=None, metrics_history=None):
    ckpt_cfg = cfg['checkpoint']

    latest_dir = ckpt_cfg['dir']
    os.makedirs(latest_dir, exist_ok=True)

    dist_dir = os.path.join(latest_dir, 'dist')
    train_dir = os.path.join(latest_dir, 'train')
    plots_dir = os.path.join(latest_dir, 'plots')

    os.makedirs(dist_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(dist_dir, 'model.pt'))

    torch.save(optimizer.state_dict(), os.path.join(train_dir, 'optimizer.pt'))
    torch.save(scheduler.state_dict(), os.path.join(train_dir, 'scheduler.pt'))
    with open(os.path.join(train_dir, 'progress.json'), 'w') as f:
        json.dump(progress, f, indent=2)

    if samples:
        with open(os.path.join(latest_dir, 'checkpoint_samples.txt'), 'w') as f:
            f.write(f"Generated samples at step {step}:\n\n")
            f.write("\n".join(samples))

    if metrics_history:
        metrics_data = {
            'step': step,
            'metrics_history': metrics_history,
            'dataset_progress': {
                'shard_index': progress.get('shard_index', 0),
                'offset': progress.get('offset', 0),
                'progress_percent': progress.get('progress_percent', 0.0)
            },
            'training_config': {
                'batch_size': cfg.get('batch_size', 0),
                'learning_rate': scheduler.get_last_lr()[0] if scheduler else 0.0,
                'total_steps_planned': cfg.get('training', {}).get('max_steps', 0)
            }
        }

        with open(os.path.join(latest_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_data, f, indent=2)

        create_checkpoint_plots(metrics_history, plots_dir, step)

    backup_interval = ckpt_cfg.get('backup_interval', 1000)
    if step % backup_interval == 0:
        backup_dir = ckpt_cfg['backups']['dir']
        os.makedirs(backup_dir, exist_ok=True)

        backup_step_dir = os.path.join(backup_dir, f'step_{step}')
        if os.path.exists(backup_step_dir):
            shutil.rmtree(backup_step_dir)
        shutil.copytree(latest_dir, backup_step_dir)

        backup_dirs = [d for d in os.listdir(backup_dir) if d.startswith(
            'step_') and os.path.isdir(os.path.join(backup_dir, d))]
        backup_dirs.sort(key=lambda x: int(x.split('_')[1]))
        keep = ckpt_cfg['backups']['keep_last']
        for old_dir in backup_dirs[:-keep]:
            shutil.rmtree(os.path.join(backup_dir, old_dir))

    if step % cfg['milestones']['interval_steps'] == 0:
        ms_dir = os.path.join(cfg['milestones']['dir'], f'step_{step}')

        if os.path.exists(ms_dir):
            shutil.rmtree(ms_dir)

        shutil.copytree(latest_dir, ms_dir)

        print(f"✅ Milestone {step}: Pure fork created at {ms_dir}")


def load_checkpoint(model, optimizer, scheduler, cfg):
    ckpt_cfg = cfg['checkpoint']
    if not cfg.get('resume', {}).get('enabled', False):
        return {}

    latest_dir = ckpt_cfg['dir']

    train_dir = os.path.join(latest_dir, 'train')
    dist_dir = os.path.join(latest_dir, 'dist')

    if os.path.exists(train_dir) and os.path.exists(dist_dir):
        model_path = os.path.join(dist_dir, 'model.pt')
        optim_path = os.path.join(train_dir, 'optimizer.pt')
        sched_path = os.path.join(train_dir, 'scheduler.pt')
        progress_path = os.path.join(train_dir, 'progress.json')
    else:
        model_path = os.path.join(latest_dir, 'model.pt')
        optim_path = os.path.join(latest_dir, 'optimizer.pt')
        sched_path = os.path.join(latest_dir, 'scheduler.pt')
        progress_path = os.path.join(latest_dir, 'progress.json')

    required_files = [model_path, optim_path, sched_path, progress_path]
    if not all(os.path.exists(p) for p in required_files):
        return {}

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    optimizer.load_state_dict(torch.load(optim_path, map_location="cpu"))
    scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))

    with open(progress_path, 'r') as f:
        progress = json.load(f)

    return progress
