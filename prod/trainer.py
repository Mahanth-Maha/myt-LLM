import os
import json
import math
import time

import shutil
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

import threading
import numpy as np
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

import wandb
from torch.utils.tensorboard import SummaryWriter

@dataclass
class TrainerConfig:
    accum_steps: int = 1                   # gradient accumulation steps
    max_grad_norm: float = 1.0             # <=0 to disable;  gradient clipping
    grad_skip_nan_inf: bool = True         # skip optimizer step if loss is NaN/Inf
    bf16_autocast: bool = True             # enable autocast for bf16/fp16

    val_every_steps: int = 1000            # run validation every               (for N optimizer steps)
    val_max_batches: int = 100             # validation #batches to use
    log_every_steps: int = 50              # printing freq
    log_grad_norm_every: int = 0           # 0=off; otherwise compute & log grad-norm every   (for N optimizer steps)
    moving_avg_alpha: float = 0.03         # smoothing for train loss 

    save_every_steps: int = 5000           # save checkpoint every              (for N optimizer steps)
    keep_last: int = 5                     # keep last K checkpoints
    async_ckpt_write: bool = True          # Async write checkpoint

    ema_decay: float = 0.0                 # 0 disables; e.g., 0.999 for EMA
    ema_update_every: int = 1              # update EMA every               (for N optimizer steps)
    
    milestone: int = 10000                 # copy that checkpoint to milestones
    resume_train: bool = False             # resume?
    train_time_warmup: int = 100           # before-training: check per iter time 
    precomputed_train_time: float = 0.0    # if already done for given bs and acc steps => set warmup = 0 and this value
    
    run_name: str = "mytslm-unk-debug"     # run name
    use_tensorboard: bool = True           # use tensorboard ?
    use_wandb: bool = False                # use wandb ?
    wandb_mode: str = "diabled"            #  "online" | "offline" | "disabled"
    wandb_project: str = "mytslm"          # wandb proj
    
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

class DecodeTransformerTrainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader,
                 device=None, log_dir="logs", scheduler=None, 
                 metric_function=None, precision_config=None,
                 trainer_config = None,  tokenizer=None, samples_cfg = None, 
                 model_kwargs=None, verbose = True):

        self.verbose = verbose
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        if self.verbose:
            print(f"💐 Using device: {self.device}")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_kwargs = model_kwargs
        self.tokenizer = tokenizer
        self.samples_cfg = samples_cfg or { 
            'prompts': ['Once upon a time'],
            'max_pred_tokens': 16,
            'temp': 1,
            'top_k': 25,
            'top_p': 0.95
        }
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

        self.tb_writer = None
        self.wandb_run = None

        self.tb_dir = os.path.join(self.log_dir, "tensorboard")
        self.wb_dir = os.path.join(self.log_dir, "wandb")
        os.makedirs(self.tb_dir, exist_ok=True)
        os.makedirs(self.wb_dir, exist_ok=True)
        
        if getattr(self.tc, "use_tensorboard", False):
            self.tb_writer = SummaryWriter(log_dir=self.tb_dir)
            
            print("✅ TensorBoard initialized.")
            print("▶️ To view logs, run this in a terminal:")
            print(f"```\ntensorboard --logdir {self.tb_dir} --port 6006\n```\n")
            print("🌐 TensorBoard logs URL: http://localhost:6006\n")
            
        if getattr(self.tc, "use_wandb", False):
            mode = getattr(self.tc, "wandb_mode", "disabled") or "disabled"
            if mode not in {"online", "offline", "disabled"}:
                mode = "offline"
            self.wandb_run = wandb.init(
                project=getattr(self.tc, "wandb_project", "myt-slm"),
                name=getattr(self.tc, "run_name", None),
                mode=mode,
                dir=self.wb_dir,
                config={
                    "trainer_config": vars(self.tc),
                    "precision": self.precision_config.get("precision", "fp32"),
                    "device": str(self.device),
                },
                resume="allow",
            )
            print(f"✅ Weights & Biases initialized in '{mode}' mode.")
            if self.wandb_run is not None:
                print("▶️ To view logs, run this in a terminal:")
                print("```\nwandb view\n```\n")
            print(f"🌐 WandB logs URL: {self.wandb_run.url}\n")

        if self.tc.resume_train:
            print('Resuming...')
            self._maybe_resume()
    
    def _log_metrics(self, step, metrics):
        clean = {}
        for k, v in metrics.items():
            if v is None:
                continue
            if isinstance(v, (int, float)):
                if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                    clean[k] = v
            else:
                pass

        if self.tb_writer is not None and clean:
            for k, v in clean.items():
                self.tb_writer.add_scalar(k, v, step)

        if self.wandb_run is not None and clean:
            wandb.log(clean, step=step)

    def _log_samples(self, step, samples, checkpoint = None):
        if self.tb_writer is not None:
            for i, (p, g) in enumerate(samples):
                text = f"**PROMPT:** {p}\n\n**GENERATED:** {g}"
                self.tb_writer.add_text(f"samples/{i}", text, step)

        if self.wandb_run is not None:
            try:
                table = wandb.Table(columns=["idx", "prompt", "generated", "checkpoint"])
                for i, (p, g) in enumerate(samples):
                    table.add_data(i, str(p), str(g), checkpoint if checkpoint is not None else step)
                wandb.log({"samples": table, "checkpoint": checkpoint or step}, step=step)
            except Exception:
                pass
    
    
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

            ctx = torch.autocast(device_type, dtype=self.autocast_dtype) if self.autocast_enabled else nullcontext()
            with ctx:
                logits = model(X)
                loss = loss_fn(logits, Y)

            total_loss += loss.item() * Y.numel()
            total_tokens += Y.numel()
            total_correct += (logits.argmax(dim=-1) == Y).sum().item()

        avg_loss = (total_loss / total_tokens) if total_tokens > 0 else float('inf')
        acc = (total_correct / total_tokens) if total_tokens > 0 else 0.0
        ppl = math.exp(avg_loss) if avg_loss < 12 else float('inf')
        bpc = avg_loss / math.log(2) if avg_loss != float('inf') else float('inf')
        model.train()
        return {"loss": avg_loss, "accuracy": acc, "perplexity": ppl, "bpc": bpc}


    def train(self, max_steps=10000, num_epochs = 1, force_epochs=False, val_iters=100, save_steps=500):
        self.model.train()
        accum = max(1, int(self.tc.accum_steps))
        print(f"🔁  Gradient accumulation: {accum} micro-steps per optimizer step")

        train_iter = iter(self.train_loader)
        steps_target = int(((num_epochs * len(self.train_loader)))/accum) if force_epochs else max_steps

        print(f'➡️  No of Epochs specified: {num_epochs} -> ~{int((num_epochs * len(self.train_loader) )/ accum)} iterations (micro-steps)')
        if not force_epochs:
            print(f"🎯  Max optimizer steps SET to: #max_steps steps  :{steps_target}")
        else:
            print(f"🎯  Max optimizer steps SET to: #num_epochs steps : {steps_target}")
        if self.tc.resume_train:
            print('Resuming from ')
            
        tokens_in_step_window = 0
        wall_start = time.time()
        loss_ma = None
        last_eval_step = self.global_step
        last_log_step = self.global_step

        with tqdm(total=steps_target - self.global_step , desc=f"OptStep {self.global_step}/{steps_target}") as pbar:
            epoch = self.start_epoch
            while self.global_step < steps_target:
                try:
                    X, Y = next(train_iter)
                except StopIteration:
                    epoch += 1
                    if not force_epochs and epoch >= num_epochs:
                        break
                    train_iter = iter(self.train_loader)
                    X, Y = next(train_iter)

                X, Y = X.to(self.device), Y.to(self.device)
                micro_tokens = Y.numel()

                ctx = torch.autocast(self.device_type, dtype=self.autocast_dtype) if (self.autocast_enabled and self.tc.bf16_autocast) else nullcontext()

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
                        if self.tc.log_grad_norm_every and (self.global_step % self.tc.log_grad_norm_every == 0):
                            grad_norm_val = clip_grad_norm_(self.model.parameters(), self.tc.max_grad_norm, error_if_nonfinite=False).item()
                        else:
                            clip_grad_norm_(self.model.parameters(), self.tc.max_grad_norm, error_if_nonfinite=False)

                    step_ok = True
                    if self.tc.grad_skip_nan_inf:
                        with torch.no_grad():
                            has_bad_grad = False
                            for p in self.model.parameters():
                                if p.grad is not None and not torch.isfinite(p.grad).all():
                                    has_bad_grad = True
                                    break
                            if has_bad_grad:
                                for p in self.model.parameters():
                                    if p.grad is not None:
                                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                                step_ok = False


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

                    # torch.cuda.synchronize() if torch.cuda.is_available() else None
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
                        val_loss = self.logs['val_loss'][-1] if self.logs["val_loss"] else None
                        val_acc  = self.logs['val_acc'][-1]  if self.logs["val_acc"]  else None
                        val_ppl  = self.logs['val_ppl'][-1]  if self.logs["val_ppl"]  else None
                        loss_diff = self.logs['loss_diff'][-1] if self.logs["loss_diff"] else None
                        grad_norm_to_log = grad_norm_val if (grad_norm_val is not None) else None

                        self._log_metrics(self.global_step, {
                            "train/loss": raw_loss,
                            "train/loss_ma": loss_ma,
                            "train/tok_per_s": tokps,
                            "train/lr": lr,
                            "train/grad_norm": grad_norm_to_log,
                            "val/loss": val_loss,
                            "val/accuracy": val_acc,
                            "val/perplexity": val_ppl,
                            "val/loss_diff": loss_diff,
                        })
                        last_log_step = self.global_step

                    pbar.update(1)

        ckpt_path = self.save_checkpoint(self.global_step, async_write=False, final_copy=True)
        print(f'✅ Final checkpoint of model saved at location: {ckpt_path}')
        
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
            print("\n\n▶️ To view TensorBoard logs, run this in a terminal:")
            print(f"```\ntensorboard --logdir {self.tb_dir} --port 6006\n```\n")
            print("\t🌐 TensorBoard logs URL: http://localhost:6006")
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
                print(f"\t🌐 WandB logs URL: {self.wandb_run.url}")
            except Exception:
                pass
        

    def warmup_time_check_for_train(self, warmup_steps=100):
        print('❔ Saving Model Check:')
        ckpt_path = self.save_checkpoint(0, async_write=False)
        print('✅ Saved Model as Checkpoint 0')
        
        print('❔ Model (forward and Backward) Check: (seems stuck? dw just wait for ~2 mins)')
        self.model.train()
        times = []
        accum = max(1, int((getattr(self.tc, "accum_steps", 1) or 1)))
        micro = 0
        start_wall = time.time()

        with tqdm(total=warmup_steps, desc="Warmup Steps") as pbar:
            for i, (x, y) in enumerate(self.train_loader):
                if i >= warmup_steps:
                    break
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)

                ctx = torch.autocast(self.device_type, dtype=self.autocast_dtype) if self.autocast_enabled else nullcontext()
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
                    # torch.cuda.synchronize() if torch.cuda.is_available() else None
                    times.append(time.time() - start_wall)
                    start_wall = time.time()
                
                pbar.update(1)
                
        print(f'✅ Model warm-up-check successful')
        
        if not times:
            print("No timing data collected.")
            self.resume(ckpt_path)
            return 0.0

        last_n = max(1, int(0.39 * len(times)))
        avg_time_recent = np.array(times[-last_n:]).mean()
        avg_time_all = np.array(times).mean()
        
        if self.verbose:
            print(f"⏲️ Time Estimates:")
            print(f"  Average time per optimizer step (all {len(times)} steps): {avg_time_all:.4f} s")
            print(f"  Average time per optimizer step (last {last_n} steps): {avg_time_recent:.4f} s [ <-- CONSIDER ]")

        print('❔ Reload Model Check:')
        self.resume(ckpt_path)
        return avg_time_recent

    def _write_ckpt(self, ckpt_path, state):
        os.makedirs(ckpt_path, exist_ok=True)
        dist_dir = os.path.join(ckpt_path, "dist")
        os.makedirs(dist_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(dist_dir, "model.pt"))
        if self.model_kwargs:
            torch.save(self.model_kwargs, os.path.join(dist_dir, "model_config.pt"))

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
        
        if step != 0:
            plots_dir = os.path.join(ckpt_path, "plots_n_samples")
            os.makedirs(plots_dir, exist_ok=True)
            self.plot_all(plots_dir)
            if self.tokenizer:
                # outputs = self.tokenizer.generate_batch(
                outputs = self.tokenizer.generate_batch_non_parallel(
                    model=self.model,
                    prompts=self.samples_cfg['prompts'],
                    max_pred_tokens=self.samples_cfg['max_pred_tokens'],
                    temp=self.samples_cfg['temp'],
                    top_k=self.samples_cfg['top_k'],
                    top_p=self.samples_cfg['top_p'],
                    device=str(self.device)
                )
                
                sample_pairs = list(zip(self.samples_cfg['prompts'], outputs))
                self._log_samples(step, sample_pairs, checkpoint=step)

                with open(os.path.join(plots_dir,"samples.txt"), "w") as f:
                    f.write(f'[{datetime.now()}] | Checkpoint: {step}\nlogs: {self.log_dir}\n')
                    f.write("="*120 + "\n")
                    f.write(f"\tInference samples from the checkpoint {step}" + "\n")
                    f.write("="*120 + "\n")
                    f.write("[Start of Samples]" + "\n")
                    it = 1
                    for p, o in zip(self.samples_cfg['prompts'], outputs):
                        f.write(f"[INFERNCE={it:2}]\n")
                        f.write(f"PROMPT:{p}\n")
                        f.write(f"GENERATED:{o}\n")
                        f.write("="*120 + "\n")
                        it += 1 
                    f.write("[End of Samples]" + "\n")
                    
        self.ckpt_history.append(ckpt_path)
        while len(self.ckpt_history) > self.tc.keep_last:
            old_ckpt = self.ckpt_history.pop(0)
            shutil.rmtree(old_ckpt, ignore_errors=True)

        if (step % self.tc.milestone == 0 or final_copy) and step != 0:
            file_name = f"milestone_{step}_final" if final_copy else f"milestone_{step}"
            milestone_path = os.path.join(self.milestones_dir, file_name)
            if os.path.exists(milestone_path):
                shutil.rmtree(milestone_path)
            shutil.copytree(ckpt_path, milestone_path)
            if final_copy:
                print(f'🥳 Final Model is saved as a milestone at : {milestone_path}')

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