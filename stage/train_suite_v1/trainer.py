import os
import torch
import json
import math
import time
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, directory, model, optimizer, loss_fn,
                 train_dataset_loader, scheduler=None,
                 device='cpu', live_plot=True):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.live_plot = live_plot

        self.directory = directory
        os.makedirs(directory, exist_ok=True)

        self.train_dataset_loader = train_dataset_loader
        self.logs = {"loss": []}
        self.last_checkpoint = 0

    def train(self, num_epochs, save_steps):
        current_checkpoint = 0
        with tqdm(total=math.ceil(len(self.train_dataset_loader) * num_epochs)) as pbar:
            for epoch in range(num_epochs):
                for batch_idx, batch in enumerate(self.train_dataset_loader):
                    pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

                    if current_checkpoint < self.last_checkpoint:
                        current_checkpoint += 1
                        pbar.update()
                        continue

                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                    loss.backward()
                    self.optimizer.step()

                    if self.scheduler:
                        self.scheduler.step()

                    self.logs["loss"].append(loss.item())

                    pbar.set_postfix({"loss": loss.item()})
                    pbar.update()
                    current_checkpoint += 1

                    if current_checkpoint % save_steps == 0:
                        self._save_checkpoint(current_checkpoint, epoch, loss.item())

                    if self.live_plot and batch_idx == 0:
                        self._update_plot(num_epochs)

            self._save_checkpoint(current_checkpoint, epoch, self.logs['loss'][-1])
            if self.live_plot:
                self._update_plot(num_epochs)

    def _save_checkpoint(self, checkpoint, epoch, loss):
        ckpt_dir = os.path.join(self.directory, f"checkpoint-{checkpoint}")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
        if self.scheduler:
            torch.save(self.scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
        with open(os.path.join(ckpt_dir, "log.json"), "w") as f:
            json.dump(self.logs, f, indent=2)

    def _update_plot(self, num_epochs):
        plt.figure(figsize=(10, 4))
        plt.plot(self.logs["loss"], label="Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss (Live)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def resume(self):
        checkpoints = [d for d in os.listdir(self.directory) if d.startswith("checkpoint-")]
        if not checkpoints:
            return
        latest_ckpt = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        self.last_checkpoint = int(latest_ckpt.split("-")[-1])
        ckpt_path = os.path.join(self.directory, latest_ckpt)
        self.model.load_state_dict(torch.load(os.path.join(ckpt_path, "model.pt"), map_location=self.device))
        self.optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, "optimizer.pt"), map_location=self.device))
        if self.scheduler and os.path.exists(os.path.join(ckpt_path, "scheduler.pt")):
            self.scheduler.load_state_dict(torch.load(os.path.join(ckpt_path, "scheduler.pt"), map_location=self.device))
        with open(os.path.join(ckpt_path, "log.json"), "r") as f:
            self.logs = json.load(f)
        print(f"Resumed from checkpoint {self.last_checkpoint}")


class GeneralTrainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader,
                 device=None, log_dir="logs", scheduler=None,
                 live_plot=True, milestone=10000, keep_last=5, metric_function = None, 
                 resume_train = True, precision_config=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        print(f"💐 Using device: {self.device}")
            
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.live_plot = live_plot
        self.milestone = milestone
        self.keep_last = keep_last
        self.resume_train = resume_train
        self.precision_config = precision_config or {
            'scaler': None,
            'autocast_enabled': False,
            'precision': 'fp32',
            'dtype': torch.float32
        }
        if metric_function:
            self.metrics_fn = metric_function
        else:
            self.metrics_fn = self._get_precision_aware_metrics

        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.ckpt_root = os.path.join(log_dir, "checkpoints")
        self.milestones_dir = os.path.join(log_dir, "milestones")

        for d in [self.ckpt_root, self.milestones_dir]:
            os.makedirs(d, exist_ok=True)

        self.logs = {
            "train_loss": [],
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
        }

        self.global_step = 0
        self.start_epoch = 0
        self.ckpt_history = []

    @torch.no_grad()
    def _get_precision_aware_metrics(self, model, val_loader, loss_fn, val_iters=100):
        model.eval()

        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        batches_processed = 0
        device_type = self.device.type
        
        try:
            val_iter = iter(val_loader)

            for i in range(val_iters):
                try:
                    X, Y = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    try:
                        X, Y = next(val_iter)
                    except StopIteration:
                        break

                X, Y = X.to(self.device), Y.to(self.device)

                if self.precision_config.get('autocast_enabled', False):
                    autocast_dtype = self.precision_config.get('dtype', torch.bfloat16)
                    with torch.amp.autocast(device_type, dtype=autocast_dtype):
                        logits = model(X)
                        loss = loss_fn(logits, Y)
                else:
                    logits = model(X)
                    loss = loss_fn(logits, Y)

                batch_loss = loss.item()
                batch_tokens = Y.numel()

                total_loss += batch_loss * batch_tokens
                total_tokens += batch_tokens

                preds = logits.argmax(dim=-1)
                total_correct += (preds == Y).sum().item()
                batches_processed += 1

        except Exception as e:
            print(f"⚠️ Error in metrics calculation: {e}")
            return self._fallback_validate()

        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            accuracy = total_correct / total_tokens
            perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
            bpc = avg_loss / math.log(2)
        else:
            avg_loss = accuracy = bpc = 0.0
            perplexity = float('inf')

        model.train()

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "perplexity": perplexity,
            "bpc": bpc,
            "batches_processed": batches_processed
        }

    @torch.no_grad()
    def _fallback_validate(self):
        self.model.eval()
        val_losses, val_accs = [], []
        device_type = self.device.type
        try:
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)

                if self.precision_config.get('autocast_enabled', False):
                    autocast_dtype = self.precision_config.get('dtype', torch.bfloat16)
                    with torch.amp.autocast(device_type, dtype=autocast_dtype):
                        preds = self.model(x)
                        loss = self.loss_fn(preds, y)
                else:
                    preds = self.model(x)
                    loss = self.loss_fn(preds, y)

                acc = (preds.argmax(dim=-1) == y).float().mean().item()
                val_losses.append(loss.item())
                val_accs.append(acc)

        except Exception as e:
            print(f"⚠️ Fallback validation error: {e}")
            return {"loss": float('inf'), "accuracy": 0.0, "perplexity": float('inf')}

        self.model.train()

        avg_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
        avg_acc = sum(val_accs) / len(val_accs) if val_accs else 0.0
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')

        return {
            "loss": avg_loss,
            "accuracy": avg_acc, 
            "perplexity": perplexity,
            "bpc": avg_loss / math.log(2)
        }

    def train(self, num_epochs, val_iters=100, save_steps=500, max_steps=10000, force_epochs=False, resume_train=True):
        self.model.train()
        if resume_train:
            ckpt_dirs = [d for d in os.listdir(self.ckpt_root) if d.startswith("checkpoint_")]
            if ckpt_dirs:
                latest_ckpt = max(ckpt_dirs, key=lambda x: int(x.split("_")[-1]))
                print(f'Resuming from {latest_ckpt}, loading model...')
                ckpt_path = os.path.join(self.ckpt_root, latest_ckpt)
                self.resume(ckpt_path)

        print(f'No of Epochs specified: {num_epochs} -> {num_epochs * len(self.train_loader)} iterations')
        if force_epochs:
            total_iters = num_epochs * len(self.train_loader)
        else:
            total_iters = min(max_steps, num_epochs * len(self.train_loader))
            print(f'Max Iterations set to : {max_steps}')
            print(f'Training with minimum : {"num_epochs" if num_epochs * len(self.train_loader) < max_steps else "max_steps" }\n(Warning: To avoid this set `force_epochs=True`)')

        print(f'Training for {total_iters}')
        val_loss, val_acc, val_ppl = float('inf'), 0, float('inf')

        with tqdm(total=total_iters) as pbar:
            for iter in range(self.start_epoch, num_epochs):
                for batch_idx, (x, y) in enumerate(self.train_loader):
                    if self.global_step >= max_steps:
                        ckpt_path = self.save_checkpoint(iter)
                        print(f'Exiting... max steps reached ({max_steps=})!\nModel stored at: {ckpt_path}')
                        return
                    self.global_step += 1
                    pbar.set_description(f"Epoch {iter+1}/{num_epochs}")

                    start_time = time.time()
                    x, y = x.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                    device_type = self.device.type
                    # OKAY you are not MAHANTH and looking in the code.... 
                    # Alright
                    # Wait a minute.... WHO ARE YOU ??? 
                    # Why are you here ??? 
                    # anyways chill out with this: https://www.youtube.com/watch?v=j5a0jTc9S10 
                    # ✌️ Peace 🕊️ 
                    if self.precision_config.get('autocast_enabled', False):
                        autocast_dtype = self.precision_config.get('dtype', torch.bfloat16)
                        with torch.amp.autocast(device_type, dtype=autocast_dtype):
                            preds = self.model(x)
                            loss = self.loss_fn(preds, y)

                        if self.precision_config.get('scaler'):
                            self.precision_config['scaler'].scale(loss).backward()
                            self.precision_config['scaler'].step(self.optimizer)
                            self.precision_config['scaler'].update()
                        else:
                            loss.backward()
                            self.optimizer.step()
                    else:
                        preds = self.model(x)
                        loss = self.loss_fn(preds, y)
                        loss.backward()
                        self.optimizer.step()

                    end_time = time.time()

                    if self.scheduler:
                        self.scheduler.step()

                    lr = self.optimizer.param_groups[0]["lr"]

                    batch_tokens = x.numel() if hasattr(x, 'numel') else x.shape[0]
                    elapsed = end_time - start_time
                    tokps = batch_tokens / elapsed if elapsed > 0 else 0

                    self.logs["train_loss"].append(loss.item())
                    self.logs["steps_train"].append(self.global_step)
                    self.logs["lr"].append(lr)
                    self.logs["tok/s"].append(tokps)

                    if self.global_step % val_iters == 0:
                        metrics = self.metrics_fn(
                            self.model, 
                            self.val_loader, 
                            self.loss_fn, 
                            val_iters=100
                        )

                        val_loss = metrics["loss"]
                        val_acc = metrics["accuracy"] 
                        val_ppl = metrics["perplexity"]
                        val_bpc = metrics.get("bpc", 0.0)

                        # Log validation metrics
                        self.logs["val_loss"].append(val_loss)
                        self.logs["val_acc"].append(val_acc)
                        self.logs["steps_val"].append(self.global_step)
                        self.logs["val_ppl"].append(val_ppl)
                        self.logs["loss_diff"].append(abs(loss.item() - val_loss))

                    pbar.set_postfix({
                        "t_loss": f"{loss.item():.4f}", 
                        "v_loss": f"{val_loss:.4f}",
                        "v_ppl": f"{val_ppl:6.2f}",
                        "v_acc": f"{val_acc:.4f}",
                        "tok/s": f"{tokps:8.0f}",
                    })
                    pbar.update()

                    if self.global_step % save_steps == 0:
                        self.save_checkpoint(iter)

        ckpt_path = self.save_checkpoint(total_iters)
        print(f'Final model saved at location: {ckpt_path}')

    def warmup_time_check_for_train(self, warmup_steps=100):
        ckpt_path = self.save_checkpoint(0)
        self.model.train()
        times = []
        for i, (x, y) in enumerate(self.train_loader):
            if i >= warmup_steps:
                break
            start = time.time()
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            if self.precision_config.get('autocast_enabled', False):
                autocast_dtype = self.precision_config.get('dtype', torch.bfloat16)
                with torch.amp.autocast('cuda', dtype=autocast_dtype):
                    preds = self.model(x)
                    loss = self.loss_fn(preds, y)

                if self.precision_config.get('scaler'):
                    self.precision_config['scaler'].scale(loss).backward()
                    self.precision_config['scaler'].step(self.optimizer)
                    self.precision_config['scaler'].update()
                else:
                    loss.backward()
                    self.optimizer.step()
            else:
                preds = self.model(x)
                loss = self.loss_fn(preds, y)
                loss.backward()
                self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()
            end = time.time()
            times.append(end - start)

        times = times[int(0.39 * warmup_steps):]
        avg_time = sum(times) / len(times) if times else 0
        self.resume(ckpt_path)
        print(f"Average time per training step over {warmup_steps} steps: {avg_time:.4f} seconds")
        return avg_time

    def save_checkpoint(self, iter):
        step = self.global_step
        ckpt_path = os.path.join(self.ckpt_root, f"checkpoint_{step}")
        os.makedirs(ckpt_path, exist_ok=True)

        dist_dir = os.path.join(ckpt_path, "dist")
        os.makedirs(dist_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(dist_dir, "model.pt"))

        train_dir = os.path.join(ckpt_path, "train")
        os.makedirs(train_dir, exist_ok=True)
        train_ckpt = {
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "logs": self.logs,
            "step": step,
            "epoch": iter,
            "precision_config": self.precision_config
        }
        torch.save(train_ckpt, os.path.join(train_dir, "train_state.pt"))

        with open(os.path.join(train_dir, "metrics.json"), "w") as f:
            json.dump(self.logs, f, indent=2)
        
        progress = {}
        dataset_progress = None
        if hasattr(self.train_loader, 'dataset') and hasattr(self.train_loader.dataset, 'get_progress_info'):
            try:
                dataset_progress = self.train_loader.dataset.get_progress_info()
                print(f"💾 Saving dataset progress: shard {dataset_progress['shard_index']}, offset {dataset_progress['offset']}")
            except Exception as e:
                print(f"⚠️ Could not get dataset progress: {e}")
                dataset_progress = None

        progress["step"] = step
        progress["epoch"] = iter
        progress["dataset_progress"] = dataset_progress
        progress["metrics_history"] = {
            "steps": self.logs.get("steps_train", []),
            "train_loss": self.logs.get("train_loss", []),
            "val_loss": self.logs.get("val_loss", []),
            "perplexity": self.logs.get("val_ppl", []),
            "accuracy": self.logs.get("val_acc", []),
            "tokens_per_second": self.logs.get("tok/s", []),
            "learning_rate": self.logs.get("lr", []),
            "timestamp": []
        }

        with open(os.path.join(train_dir, "progress.json"), "w") as f:
            json.dump(progress, f, indent=2)

        plots_dir = os.path.join(ckpt_path, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        self.plot_all(plots_dir)

        self.ckpt_history.append(ckpt_path)
        if len(self.ckpt_history) > self.keep_last:
            old_ckpt = self.ckpt_history.pop(0)
            shutil.rmtree(old_ckpt, ignore_errors=True)

        if step % self.milestone == 0:
            milestone_path = os.path.join(self.milestones_dir, f"milestone_{step}")
            if os.path.exists(milestone_path):
                shutil.rmtree(milestone_path)
            shutil.copytree(ckpt_path, milestone_path)

        return ckpt_path

    def resume(self, ckpt_path):
        train_ckpt = torch.load(os.path.join(ckpt_path, "train/train_state.pt"), map_location=self.device)
        self.global_step = train_ckpt.get("step", 0)
        self.start_epoch = train_ckpt.get("epoch", 0)
        self.logs = train_ckpt.get("logs", self.logs)

        if "precision_config" in train_ckpt:
            self.precision_config = train_ckpt["precision_config"]

        self.model.load_state_dict(torch.load(
            os.path.join(ckpt_path, "dist/model.pt"), map_location=self.device
        ))
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
            print(f"⚠️ Could not resume dataset progress: {e}")

        print(f"✅ Resumed from {ckpt_path} at step {self.global_step}, epoch {self.start_epoch}")

    def plot_losses(self, save_dir=None):
        plt.figure(figsize=(8, 5))
        plt.plot(self.logs["steps_train"], self.logs["train_loss"], label="Train Loss")
        plt.plot(self.logs["steps_val"], self.logs["val_loss"], label="Val Loss")
        plt.legend(); plt.grid(); plt.title("Loss")
        if save_dir: plt.savefig(os.path.join(save_dir, "loss.png")); plt.close()
        else: plt.show()

    def plot_accuracy(self, save_dir=None):
        plt.figure(figsize=(8, 5))
        # plt.plot(self.logs["steps_train"], self.logs["train_acc"], label="Train Acc")
        plt.plot(self.logs["steps_val"], self.logs["val_acc"], label="Val Acc")
        plt.legend(); plt.grid(); plt.title("Accuracy")
        if save_dir: plt.savefig(os.path.join(save_dir, "accuracy.png")); plt.close()
        else: plt.show()

    def plot_perplexity(self, save_dir=None):
        plt.figure(figsize=(8, 5))
        # plt.plot(self.logs["steps_train"], self.logs["train_ppl"], label="Train PPL")
        plt.plot(self.logs["steps_val"], self.logs["val_ppl"], label="Val PPL")
        plt.legend(); plt.grid(); plt.title("Perplexity")
        if save_dir: plt.savefig(os.path.join(save_dir, "ppl.png")); plt.close()
        else: plt.show()

    def plot_lr(self, save_dir=None):
        plt.figure(figsize=(8, 5))
        plt.plot(self.logs["steps_train"], self.logs["lr"], label="Learning Rate")
        plt.legend(); plt.grid(); plt.title("Learning Rate")
        if save_dir: plt.savefig(os.path.join(save_dir, "lr.png")); plt.close()
        else: plt.show()

    def plot_loss_diff(self, save_dir=None):
        plt.figure(figsize=(8, 5))
        plt.plot(self.logs["steps_val"], self.logs["loss_diff"], label="|Train - Val|")
        plt.legend(); plt.grid(); plt.title("Loss Difference")
        if save_dir: plt.savefig(os.path.join(save_dir, "loss_diff.png")); plt.close()
        else: plt.show()

    def plot_all(self, save_dir):
        self.plot_losses(save_dir)
        self.plot_accuracy(save_dir)
        self.plot_perplexity(save_dir)
        self.plot_lr(save_dir)
        self.plot_loss_diff(save_dir)

    def get_logs(self):
        return self.logs