import os
import json
import math
import time
import shutil
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import threading
import numpy as np
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_


@dataclass
class TrainerConfig:
    accum_steps: int = 1                   # gradient accumulation steps
    max_grad_norm: float = 1.0             # gradient clipping; set <=0 to disable
    grad_skip_nan_inf: bool = True         # skip optimizer step if loss is NaN/Inf
    bf16_autocast: bool = True             # enable autocast for bf16/fp16 (uses dtype from precision_config)
    log_grad_norm_every: int = 0           # 0=off; otherwise compute & log grad-norm every N optimizer steps

    val_every_steps: int = 1000            # run validation every N optimizer steps
    val_max_batches: int = 100             # how many batches to use during validation
    log_every_steps: int = 50              # update tqdm postfix + store moving averages
    moving_avg_alpha: float = 0.03         # smoothing for train loss

    save_every_steps: int = 5000           # save checkpoint every N optimizer steps
    keep_last: int = 5                     # keep last K checkpoints
    async_ckpt_write: bool = True          # write checkpoint in a background thread

    ema_decay: float = 0.0                 # 0 disables; e.g., 0.999 for EMA
    ema_update_every: int = 1              # update EMA every N optimizer steps
    
    milestone: int = 10000                 # copy that checkpoint to milestones
    resume_train: bool = False             # resume?
    train_time_warmup: int = 100           # to check time 
    

class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=(1.0 - d))

    def store(self, model):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def copy_to(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                v.data.copy_(self.shadow[k].data)

    @torch.no_grad()
    def restore(self, model):
        for k, v in model.state_dict().items():
            if k in self.backup and v.dtype.is_floating_point:
                v.data.copy_(self.backup[k].data)
        self.backup = None

class GeneralTrainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader,
                 device=None, log_dir="logs", scheduler=None, 
                 metric_function=None, precision_config=None,
                 trainer_config = None, verbose = True):

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.verbose = verbose
        if self.verbose:
            print(f"💐 Using device: {self.device}")

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.precision_config = precision_config or {
            'scaler': None,
            'autocast_enabled': False,
            'precision': 'fp32',
            'dtype': torch.float32
        }
        self.scaler = self.precision_config.get('scaler', None)
        self.autocast_enabled = self.precision_config.get('autocast_enabled', False)
        self.autocast_dtype = self.precision_config.get('dtype', torch.bfloat16)
        self.device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'

        self.metrics_fn = metric_function if metric_function else self._get_precision_aware_metrics

        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.ckpt_root = os.path.join(log_dir, "checkpoints")
        self.milestones_dir = os.path.join(log_dir, "milestones")
        for d in [self.ckpt_root, self.milestones_dir]:
            os.makedirs(d, exist_ok=True)

        self.logs = {
            "train_loss": [],
            "train_loss_ma": [],
            "val_loss": [],
            "steps_train": [],
            "steps_val": [],
            "train_acc": [],
            "val_acc": [],
            "train_ppl": [],
            "val_ppl": [],
            "lr": [],
            "loss_diff": [],
            "tok/s": [],
            "grad_norm": [],
        }

        self.global_step = 0
        self.micro_step = 0
        self.start_epoch = 0
        self.ckpt_history = []

        self.tc = trainer_config or TrainerConfig()

        self.ema = EMA(self.model, self.tc.ema_decay) if self.tc.ema_decay > 0.0 else None

        if self.tc.resume_train:
            self._maybe_resume()

    @torch.inference_mode()
    def _get_precision_aware_metrics(self, model, val_loader, loss_fn, val_iters=100):
        model.eval()
        total_loss, total_correct, total_tokens = 0.0, 0, 0
        device_type = self.device_type
        it = iter(val_loader)

        for _ in range(val_iters):
            try:
                X, Y = next(it)
            except StopIteration:
                break
            X, Y = X.to(self.device), Y.to(self.device)

            ctx = torch.amp.autocast(device_type, dtype=self.autocast_dtype) if self.autocast_enabled else nullcontext()
            with ctx:
                logits = model(X)
                loss = loss_fn(logits, Y)

            total_loss += loss.item() * Y.numel()
            total_tokens += Y.numel()
            total_correct += (logits.argmax(dim=-1) == Y).sum().item()

        avg_loss = (total_loss / total_tokens) if total_tokens > 0 else float('inf')
        acc = (total_correct / total_tokens) if total_tokens > 0 else 0.0
        ppl = math.exp(avg_loss) if avg_loss < 10 else float('inf')
        bpc = avg_loss / math.log(2) if avg_loss != float('inf') else float('inf')
        model.train()
        return {"loss": avg_loss, "accuracy": acc, "perplexity": ppl, "bpc": bpc}


    def train(self, max_steps=10000, num_epochs = 1, force_epochs=False, val_iters=100, save_steps=500):
        """
        NOTE: self.global_step counts OPTIMIZER STEPS (after accum_steps micro-steps)
        """        
        self.model.train()
        accum = max(1, int(self.tc.accum_steps))
        print(f"🔁 Gradient accumulation: {accum} micro-steps per optimizer step")

        train_iter = iter(self.train_loader)
        steps_target = (num_epochs * len(self.train_loader)) if force_epochs else max_steps

        print(f'No of Epochs specified: {num_epochs} -> ~{num_epochs * len(self.train_loader)} iterations (micro-steps)')
        if not force_epochs:
            print(f"Max optimizer steps set to: {max_steps}")

        tokens_in_step_window = 0
        wall_start = time.time()
        loss_ma = None
        last_eval_step = self.global_step
        last_log_step = self.global_step

        with tqdm(total=steps_target, desc=f"OptStep {self.global_step}/{steps_target}") as pbar:
            epoch = self.start_epoch
            while self.global_step < steps_target:
                try:
                    X, Y = next(train_iter)
                except StopIteration:
                    epoch += 1
                    if epoch >= num_epochs and not force_epochs:
                        break
                    train_iter = iter(self.train_loader)
                    continue

                X, Y = X.to(self.device), Y.to(self.device)
                micro_tokens = Y.numel()

                ctx = torch.amp.autocast(self.device_type, dtype=self.autocast_dtype) if (self.autocast_enabled and self.tc.bf16_autocast) else nullcontext()

                with ctx:
                    logits = self.model(X)
                    loss = self.loss_fn(logits, Y) / accum

                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                self.micro_step += 1
                tokens_in_step_window += micro_tokens

                raw_loss = loss.item() * accum
                if loss_ma is None:
                    loss_ma = raw_loss
                else:
                    loss_ma = (1 - self.tc.moving_avg_alpha) * loss_ma + self.tc.moving_avg_alpha * raw_loss

                if self.micro_step % accum == 0:
                    grad_norm_val = None
                    if self.tc.max_grad_norm and self.tc.max_grad_norm > 0:
                        if self.scaler:
                            self.scaler.unscale_(self.optimizer)
                        grad_norm_val = clip_grad_norm_(self.model.parameters(), self.tc.max_grad_norm).item()

                    step_ok = True
                    if self.tc.grad_skip_nan_inf:
                        if any(torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                               for p in self.model.parameters() if p.grad is not None):
                            step_ok = False
                            for p in self.model.parameters():
                                if p.grad is not None:
                                    p.grad.detach_()
                                    p.grad.zero_()

                    if step_ok:
                        if self.scaler:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        if self.scheduler:
                            self.scheduler.step()
                    else:
                        self.optimizer.zero_grad(set_to_none=True)

                    self.global_step += 1
                    self.micro_step = 0

                    if self.ema and (self.global_step % self.tc.ema_update_every == 0):
                        self.ema.update(self.model)

                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    now = time.time()
                    elapsed = now - wall_start
                    tokps = (tokens_in_step_window / elapsed) if elapsed > 0 else 0.0
                    wall_start = now
                    self.logs["tok/s"].append(tokps)
                    tokens_in_step_window = 0

                    lr = self.optimizer.param_groups[0]["lr"]
                    self.logs["lr"].append(lr)
                    self.logs["train_loss"].append(raw_loss)
                    self.logs["train_loss_ma"].append(loss_ma)
                    self.logs["steps_train"].append(self.global_step)
                    if grad_norm_val is not None and (self.tc.log_grad_norm_every and (self.global_step % self.tc.log_grad_norm_every == 0)):
                        self.logs["grad_norm"].append(grad_norm_val)
                    else:
                        self.logs["grad_norm"].append(None)

                    need_val = (self.global_step - last_eval_step) >= self.tc.val_every_steps
                    if need_val:
                        if self.ema:
                            self.ema.store(self.model)
                            self.ema.copy_to(self.model)
                        metrics = self.metrics_fn(self.model, self.val_loader, self.loss_fn, val_iters=self.tc.val_max_batches)
                        if self.ema:
                            self.ema.restore(self.model)

                        val_loss = metrics["loss"]
                        val_acc = metrics["accuracy"]
                        val_ppl = metrics["perplexity"]
                        val_bpc = metrics.get("bpc", 0.0)

                        self.logs["val_loss"].append(val_loss)
                        self.logs["val_acc"].append(val_acc)
                        self.logs["steps_val"].append(self.global_step)
                        self.logs["val_ppl"].append(val_ppl)
                        self.logs["loss_diff"].append(abs(loss_ma - val_loss))
                        last_eval_step = self.global_step

                    if self.global_step % self.tc.save_every_steps == 0:
                        self.save_checkpoint(self.global_step, async_write=self.tc.async_ckpt_write)

                    if (self.global_step - last_log_step) >= self.tc.log_every_steps or need_val:
                        pbar.set_description(f"OptStep {self.global_step}/{steps_target}")
                        pbar.set_postfix({
                            "t_loss": f"{raw_loss:.4f}",
                            "t_ma": f"{loss_ma:.4f}",
                            "v_loss": f"{self.logs['val_loss'][-1]:.4f}" if self.logs["val_loss"] else "—",
                            "tok/s": f"{tokps:8.0f}",
                            "lr": f"{lr:.2e}",
                        })
                        last_log_step = self.global_step

                    pbar.update(1)

        ckpt_path = self.save_checkpoint(self.global_step, async_write=False, final_copy=True)
        print(f'Final model saved at location: {ckpt_path}')

    def warmup_time_check_for_train(self, warmup_steps=100):
        ckpt_path = self.save_checkpoint(0, async_write=True)
        self.model.train()
        times = []
        accum = max(1, int((getattr(self.tc, "accum_steps", 1) or 1)))
        micro = 0
        start_wall = time.time()

        for i, (x, y) in enumerate(self.train_loader):
            if i >= warmup_steps:
                break
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            ctx = torch.amp.autocast(self.device_type, dtype=self.autocast_dtype) if self.autocast_enabled else nullcontext()
            with ctx:
                preds = self.model(x)
                loss = self.loss_fn(preds, y) / accum

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            micro += 1
            if micro % accum == 0:
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler:
                    self.scheduler.step()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                times.append(time.time() - start_wall)
                start_wall = time.time()

        if not times:
            print("No timing data collected.")
            self.resume(ckpt_path)
            return 0.0

        last_n = max(1, int(0.39 * len(times)))
        avg_time_recent = np.array(times[-last_n:]).mean()
        avg_time_all = np.array(times).mean()
        
        if self.verbose:
            print(f"Average time per optimizer step (all {len(times)} steps): {avg_time_all:.4f} s")
            print(f"Average time per optimizer step (last {last_n} steps): {avg_time_recent:.4f} s [ <-- CONSIDER ]")

        self.resume(ckpt_path)
        return avg_time_recent

    def _write_ckpt(self, ckpt_path, state):
        os.makedirs(ckpt_path, exist_ok=True)
        dist_dir = os.path.join(ckpt_path, "dist")
        os.makedirs(dist_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(dist_dir, "model.pt"))

        train_dir = os.path.join(ckpt_path, "train")
        os.makedirs(train_dir, exist_ok=True)
        torch.save(state, os.path.join(train_dir, "train_state.pt"))

        with open(os.path.join(train_dir, "metrics.json"), "w") as f:
            json.dump(self.logs, f, indent=2)

    def save_checkpoint(self, step_or_iter, async_write=True, final_copy = False):
        step = self.global_step if isinstance(step_or_iter, int) else step_or_iter
        ckpt_path = os.path.join(self.ckpt_root, f"checkpoint_{step}")
        state = {
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "logs": self.logs,
            "step": self.global_step,
            "epoch": self.start_epoch,
            "precision_config": self.precision_config
        }

        if async_write:
            t = threading.Thread(target=self._write_ckpt, args=(ckpt_path, state), daemon=True)
            t.start()
        else:
            self._write_ckpt(ckpt_path, state)

        plots_dir = os.path.join(ckpt_path, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        self.plot_all(plots_dir)
        
        self.ckpt_history.append(ckpt_path)
        while len(self.ckpt_history) > self.tc.keep_last:
            old_ckpt = self.ckpt_history.pop(0)
            shutil.rmtree(old_ckpt, ignore_errors=True)

        if step % self.tc.milestone == 0 :
            milestone_path = os.path.join(self.milestones_dir, f"milestone_{step}")
            if os.path.exists(milestone_path):
                shutil.rmtree(milestone_path)
            shutil.copytree(ckpt_path, milestone_path)
        if final_copy:
            milestone_path = os.path.join(self.milestones_dir, f"milestone_{step}_final")
            if os.path.exists(milestone_path):
                shutil.rmtree(milestone_path)
            shutil.copytree(ckpt_path, milestone_path)

        return ckpt_path

    def _maybe_resume(self):
        ckpt_dirs = [d for d in os.listdir(self.ckpt_root) if d.startswith("checkpoint_")] if os.path.exists(self.ckpt_root) else []
        if not ckpt_dirs:
            return
        latest_ckpt = max(ckpt_dirs, key=lambda x: int(x.split("_")[-1]))
        if self.verbose:
            print(f'Resuming from {latest_ckpt}, loading model...')
        ckpt_path = os.path.join(self.ckpt_root, latest_ckpt)
        self.resume(ckpt_path)

    def resume(self, ckpt_path):
        train_ckpt = torch.load(os.path.join(ckpt_path, "train/train_state.pt"), map_location=self.device)
        self.global_step = train_ckpt.get("step", 0)
        self.start_epoch = train_ckpt.get("epoch", 0)
        self.logs = train_ckpt.get("logs", self.logs)

        if "precision_config" in train_ckpt:
            self.precision_config = train_ckpt["precision_config"]
            self.scaler = self.precision_config.get('scaler', None)
            self.autocast_enabled = self.precision_config.get('autocast_enabled', False)
            self.autocast_dtype = self.precision_config.get('dtype', torch.bfloat16)

        self.model.load_state_dict(torch.load(os.path.join(ckpt_path, "dist/model.pt"), map_location=self.device))
        self.model = self.model.to(self.device)

        self.optimizer.load_state_dict(train_ckpt["optimizer_state"])
        if self.scheduler and train_ckpt["scheduler_state"]:
            self.scheduler.load_state_dict(train_ckpt["scheduler_state"])


        try:
            progress_file = os.path.join(ckpt_path, "train/progress.json")
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                dataset_progress = progress_data.get("dataset_progress")
                if dataset_progress and hasattr(self.train_loader, 'dataset'):
                    if hasattr(self.train_loader.dataset, 'current_shard_idx'):
                        self.train_loader.dataset.current_shard_idx = dataset_progress.get('shard_index', 0)
                        self.train_loader.dataset.current_offset = dataset_progress.get('offset', 0)
                        print(f"🔄 Resumed dataset from shard {dataset_progress['shard_index']}, offset {dataset_progress['offset']}")
                        print(f"📊 Dataset progress: {dataset_progress.get('progress_percent', 0):.2f}%")
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Could not resume dataset progress: {e}")

        if self.verbose:
            print(f"✅ Resumed from {ckpt_path} at step {self.global_step}, epoch {self.start_epoch}")

    def plot_losses(self, save_dir=None):
        plt.figure(figsize=(8,5))
        plt.plot(self.logs["steps_train"], self.logs["train_loss"], label="Train Loss")
        if self.logs["train_loss_ma"]:
            plt.plot(self.logs["steps_train"], self.logs["train_loss_ma"], label="Train Loss (MA)")
        plt.plot(self.logs["steps_val"], self.logs["val_loss"], label="Val Loss")
        plt.legend(); plt.grid(); plt.title("Loss")
        if save_dir: 
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, "loss.png")); plt.close()
        else: 
            plt.show()

    def plot_accuracy(self, save_dir=None):
        plt.figure(figsize=(8,5))
        if self.logs["val_acc"]:
            plt.plot(self.logs["steps_val"], self.logs["val_acc"], label="Val Acc")
        plt.legend(); plt.grid(); plt.title("Accuracy")
        if save_dir: 
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, "accuracy.png")); plt.close()
        else: 
            plt.show()

    def plot_perplexity(self, save_dir=None):
        plt.figure(figsize=(8,5))
        if self.logs["val_ppl"]:
            plt.plot(self.logs["steps_val"], self.logs["val_ppl"], label="Val PPL")
        plt.legend(); plt.grid(); plt.title("Perplexity")
        if save_dir: 
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, "ppl.png")); plt.close()
        else: 
            plt.show()

    def plot_lr(self, save_dir=None):
        plt.figure(figsize=(8,5))
        plt.plot(self.logs["steps_train"], self.logs["lr"], label="Learning Rate")
        plt.legend(); plt.grid(); plt.title("Learning Rate")
        if save_dir: 
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, "lr.png")); plt.close()
        else: 
            plt.show()

    def plot_loss_diff(self, save_dir=None):
        plt.figure(figsize=(8,5))
        if self.logs["loss_diff"]:
            plt.plot(self.logs["steps_val"], self.logs["loss_diff"], label="|TrainMA - Val|")
        plt.legend(); plt.grid(); plt.title("Loss Difference")
        if save_dir: 
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, "loss_diff.png")); plt.close()
        else: 
            plt.show()

    def plot_all(self, save_dir):
        self.plot_losses(save_dir)
        self.plot_accuracy(save_dir)
        self.plot_perplexity(save_dir)
        self.plot_lr(save_dir)
        self.plot_loss_diff(save_dir)

    def get_logs(self):
        return self.logs