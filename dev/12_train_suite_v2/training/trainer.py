import time
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.logging import setup_logging
from utils.checkpointing import save_checkpoint, load_checkpoint
from utils.metrics import compute_loss_perplexity, compute_accuracy, tokens_per_second


class Trainer:
    def __init__(self, cfg, model, dataset, optimizer, scheduler, val_dataset=None):
        self.cfg = cfg
        self.device = torch.device(cfg.get('device', 'cpu'))
        self.model = model.to(self.device)
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = setup_logging(cfg, name='train')
        self.start_step = 0

        self.metrics_history = {
            'steps': [],
            'train_loss': [],
            'val_loss': [],
            'perplexity': [],
            'accuracy': [],
            'tokens_per_second': [],
            'learning_rate': [],
            'timestamp': []
        }

        self.current_metrics = {
            'train_loss': 0.0,
            'val_loss': 0.0,
            'perplexity': 0.0,
            'accuracy': 0.0,
            'tokens_per_second': 0.0
        }

        progress = load_checkpoint(
            self.model, self.optimizer, self.scheduler, cfg)
        self.start_step = progress.get('step', 0)

        if 'metrics_history' in progress:
            self.metrics_history = progress['metrics_history']

        self.dataset.resume_info = progress
        self.batch_size = cfg['batch_size']

        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=__import__('data.collate', fromlist=[
                                  'collate_fn']).collate_fn,
            num_workers=cfg['dataloader_settings']['num_workers'],
            pin_memory=cfg['dataloader_settings']['pin_memory']
        )

    def validate(self):
        """Run validation and return metrics"""
        if self.val_dataset is None:
            return self.current_metrics['val_loss'], self.current_metrics['perplexity'], self.current_metrics['accuracy']

        self.model.eval()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size)
        max_val_batches = 10

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= max_val_batches:
                    break

                inputs = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = self.model(inputs)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100
                )

                acc = compute_accuracy(logits, labels)

                total_loss += loss.item()
                total_acc += acc
                num_batches += 1

        self.model.train()

        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_acc = total_acc / num_batches
            ppl = compute_loss_perplexity(avg_loss)
            return avg_loss, ppl, avg_acc
        else:
            return self.current_metrics['val_loss'], self.current_metrics['perplexity'], self.current_metrics['accuracy']

    def generate_samples(self, num_samples=3):
        self.model.eval()
        samples = []

        prompts = [
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            torch.tensor([[50, 100, 150]], dtype=torch.long),
            torch.tensor([[1000, 2000]], dtype=torch.long)
        ]

        for i, prompt in enumerate(prompts[:num_samples]):
            try:
                with torch.no_grad():
                    output = self.model.generate(
                        prompt.to(self.device),
                        max_pred_tokens=50,
                        temp=0.8,
                        top_k=40,
                        top_p=0.9
                    )
                    samples.append(
                        f"Sample {i+1}: {output.squeeze().tolist()}")
            except Exception as e:
                samples.append(f"Sample {i+1}: [Generation failed: {str(e)}]")

        self.model.train()
        return samples

    def update_metrics_history(self, step, train_loss, val_loss, ppl, acc, tps, lr, timestamp):
        self.metrics_history['steps'].append(step)
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['perplexity'].append(ppl)
        self.metrics_history['accuracy'].append(acc)
        self.metrics_history['tokens_per_second'].append(tps)
        self.metrics_history['learning_rate'].append(lr)
        self.metrics_history['timestamp'].append(timestamp)

        self.current_metrics.update({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'perplexity': ppl,
            'accuracy': acc,
            'tokens_per_second': tps
        })

    def pretrain_check(self, num_iters=10):
        self.logger.info("Running pretrain check...")
        iterator = iter(self.dataloader)

        _ = next(iterator)
        times = []
        for i in range(num_iters):
            start = time.time()
            batch = next(iterator)
            inputs, labels = batch['input_ids'].to(
                self.device), batch['labels'].to(self.device)
            loss = self._forward_backward(inputs, labels)
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            elapsed = time.time() - start
            if i > 0:
                times.append(elapsed)
            self.logger.info(
                f"Pretrain iter {i+1}/{num_iters}, loss={loss:.4f}, time={elapsed:.3f}s")

        avg_time = sum(times) / len(times)
        eta = avg_time * (len(self.dataloader) - num_iters)
        self.logger.info(
            f"Average per iteration time (batch_size={self.batch_size}): {avg_time:.4f}s")
        self.logger.info(f"No of iterations per epoch: {len(self.dataloader)}")
        self.logger.info(f"Estimated full epoch time: {eta/3600:.2f}h")
        return avg_time

    def _forward_backward(self, inputs, labels):
        self.model.train()
        logits = self.model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.get('gradient_clipping', 1.0))
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def train_loop(self):
        avg_pretrain = self.pretrain_check()
        self.logger.info(
            f"Starting Training loop\nUsing avg iteration time: {avg_pretrain:.3f}s")

        total_steps = self.start_step
        pbar = tqdm(self.dataloader, initial=self.start_step,
                    total=self.cfg['training']['max_steps'], desc="Training")

        for batch in pbar:
            inputs = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            train_loss = self._forward_backward(inputs, labels)
            total_steps += 1

            tps = tokens_per_second(inputs.numel(), avg_pretrain)
            lr = self.scheduler.get_last_lr()[0]

            self.current_metrics['train_loss'] = train_loss
            self.current_metrics['tokens_per_second'] = tps

            eval_interval = self.cfg['training'].get('eval_interval', 1000)
            if total_steps % eval_interval == 0:
                val_loss, ppl, acc = self.validate()
                self.current_metrics.update({
                    'val_loss': val_loss,
                    'perplexity': ppl,
                    'accuracy': acc
                })

            if total_steps % self.cfg['logging']['interval_steps'] == 0:
                self.logger.info(
                    f"Step {total_steps}: "
                    f"train_loss={self.current_metrics['train_loss']:.4f}, "
                    f"val_loss={self.current_metrics['val_loss']:.4f}, "
                    f"ppl={self.current_metrics['perplexity']:.2f}, "
                    f"acc={self.current_metrics['accuracy']:.3f}, "
                    f"tps={self.current_metrics['tokens_per_second']:.0f}, "
                    f"lr={lr:.2e}"
                )

                timestamp = time.time()
                self.update_metrics_history(
                    total_steps, train_loss, self.current_metrics['val_loss'],
                    self.current_metrics['perplexity'], self.current_metrics['accuracy'],
                    tps, lr, timestamp
                )

            checkpoint_interval = self.cfg['checkpoint'].get(
                'save_interval', 1000)
            if total_steps % checkpoint_interval == 0:
                self.logger.info(
                    f"🔄 Checkpointing at step {total_steps} (with real-time plots)...")

                samples = self.generate_samples()
                self.logger.info("Generated samples:")
                for sample in samples:
                    self.logger.info(f"  {sample}")

                progress = {
                    'step': total_steps,
                    'metrics_history': self.metrics_history,
                    **self.dataset.get_progress_info()
                }

                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    progress, self.cfg, total_steps, samples, self.metrics_history
                )

                self.logger.info(
                    f"✅ Checkpoint saved with real-time plots at step {total_steps}")

            if total_steps >= self.cfg['training']['max_steps']:
                break

            pbar.set_postfix({
                'loss': f"{self.current_metrics['train_loss']:.3f}",
                'ppl': f"{self.current_metrics['perplexity']:.1f}",
                'acc': f"{self.current_metrics['accuracy']:.3f}",
                'tps': f"{self.current_metrics['tokens_per_second']:.0f}"
            })

        self.logger.info("Final checkpoint...")
        samples = self.generate_samples()
        progress = {
            'step': total_steps,
            'metrics_history': self.metrics_history,
            **self.dataset.get_progress_info()
        }
        save_checkpoint(
            self.model, self.optimizer, self.scheduler,
            progress, self.cfg, total_steps, samples, self.metrics_history
        )
