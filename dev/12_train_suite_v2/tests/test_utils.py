import os
import tempfile
import torch
import pytest

from utils.logging import setup_logging
from utils.metrics import compute_loss_perplexity, compute_accuracy, tokens_per_second
from utils.sampling import top_k_top_p_sampling
from utils.checkpointing import save_checkpoint, load_checkpoint

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

LOG_DIR = "./checkpoints/logs/tests"
CKPT_DIR = tempfile.mkdtemp()

os.makedirs(LOG_DIR, exist_ok=True)


def test_setup_logging_creates_file_and_console():
    cfg = {
        'logging': {
            'log_files_dir': LOG_DIR,
            'train_log_file': os.path.join(LOG_DIR, 'train.log')
        }
    }
    logger = setup_logging(cfg, name='train')
    logger.info("Test log message")

    log_file = os.path.join(LOG_DIR, 'train.log')
    assert os.path.isfile(log_file)
    with open(log_file, 'r') as f:
        content = f.read()
    assert "Test log message" in content


def test_compute_loss_perplexity_and_accuracy_and_tps():
    loss = 1.5
    ppl = compute_loss_perplexity(loss)
    assert pytest.approx(ppl, rel=1e-5) == torch.exp(torch.tensor(loss)).item()

    logits = torch.tensor([[[2.0, 0.5], [0.1, 0.9]]])
    labels = torch.tensor([[0, 1]])
    acc = compute_accuracy(logits, labels)
    assert acc == 1.0

    tps = tokens_per_second(1000, 2.0)
    assert tps == 500.0


def test_top_k_top_p_sampling_shapes():
    logits = torch.randn(3, 100)

    out = top_k_top_p_sampling(logits, temperature=0.7)
    assert out.shape == (3, 1)

    out_k = top_k_top_p_sampling(logits, top_k=5)
    assert out_k.shape == (3, 1)

    out_p = top_k_top_p_sampling(logits, top_p=0.9)
    assert out_p.shape == (3, 1)

    out_kp = top_k_top_p_sampling(logits, top_k=5, top_p=0.8, temperature=1.2)
    assert out_kp.shape == (3, 1)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.lin(x)


def test_save_and_load_checkpoint(tmp_path):

    model = DummyModel()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 0.95)

    progress = {'step': 10, 'shard_index': 0, 'offset': 0}

    cfg = {
        'checkpoint': {
            'dir': str(tmp_path / 'latest'),
            'backups': {'dir': str(tmp_path / 'backups'), 'keep_last': 2}
        },
        'milestones': {'dir': str(tmp_path / 'milestones'), 'interval_steps': 5},
        'resume': {'enabled': True}
    }
    save_checkpoint(model, optimizer, scheduler, progress, cfg, step=10)

    for p in model.parameters():
        p.data.add_(1.0)

    loaded_progress = load_checkpoint(model, optimizer, scheduler, cfg)

    assert loaded_progress == progress

    for p in model.parameters():
        assert torch.all(p.data != 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
