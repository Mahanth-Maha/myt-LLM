import json
import torch
import pytest
from pathlib import Path
# from omegaconf import OmegaConf
# from types import SimpleNamespace
from box import Box

from data.streaming_dataset import StreamingTextDataset
from model.architecture import DecoderOnlyTransformer
from model.optimizer_config import get_optimizer, get_scheduler
from training.trainer import Trainer


# def dict_to_namespace(d):
#     if isinstance(d, dict):
#         return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
#     elif isinstance(d, list):
#         return [dict_to_namespace(x) for x in d]
#     return d

TEMP_VOCAB_SIZE = 1000


def create_dummy_shards(root: Path, num_shards: int = 2, tokens_per_shard: int = 300, vocab_size=300):
    part = root / "part-000"
    part.mkdir(parents=True, exist_ok=True)
    for i in range(num_shards):
        tokens = torch.randint(
            0, vocab_size, (tokens_per_shard,), dtype=torch.long)
        torch.save(tokens, part / f"shard_{i:05d}.pt")
    (root / "metadata.json").write_text("{}")


@pytest.fixture
def test_env(tmp_path, monkeypatch):
    base = tmp_path / "checkpoints"
    dirs = {
        'latest': base / "latest",
        'backups': base / "backups",
        'milestones': base / "milestones",
        'logs': base / "logs"
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    shards = tmp_path / "data" / "shards"
    create_dummy_shards(shards, vocab_size=TEMP_VOCAB_SIZE)
    cfg = {
        'device': 'cpu',
        'batch_size': 2,
        'logging': {'interval_steps': 1,
                    'log_files_dir': str(dirs['logs']),
                    'train_log_file': str(dirs['logs'] / "train.log")},
        'dataloader_settings': {'num_workers': 0, 'pin_memory': False},
        'gradient_clipping': 1.0,
        'scheduler': {'warmup_steps': 1},
        'optimizer': {'lr': 1e-3, 'weight_decay': 0.0},
        'checkpoint': {
            'dir': str(dirs['latest']),
            'backups': {'dir': str(dirs['backups']), 'keep_last': 2}
        },
        'milestones': {'dir': str(dirs['milestones']), 'interval_steps': 2, 'dist_subfolder': 'dist'},
        'resume': {'enabled': True, 'progress_file': str(dirs['latest'] / "progress.json")},
        "training": {"use_checkpoint": False, "max_steps": 2},
    }
    # OmegaConf for attribute access
    # cfg = OmegaConf.create(cfg)
    # cfg = dict_to_namespace(cfg)
    cfg = Box(cfg, default_box=True, box_dots=True)
    return tmp_path, shards, cfg, dirs


def build_tiny_model(cfg):
    return DecoderOnlyTransformer(
        vocab_size=TEMP_VOCAB_SIZE,
        context_length=128,
        model_dimension=64,
        n_heads=4,
        Nx_blocks=2,
        ffn_hid_dim=256,
        non_linearity='gelu',
        dropout=0.0,
        tie_weights=True,
        use_checkpoint=False
    )


def test_full_train_and_artifacts(test_env):
    tmp_path, shards, cfg, dirs = test_env

    ds = StreamingTextDataset(str(shards), context_length=128, shard_size=300)

    model = build_tiny_model(cfg)
    optim = get_optimizer(model, cfg)
    scheduler = get_scheduler(optim, cfg, total_steps=cfg.training.max_steps)

    trainer = Trainer(cfg, model, ds, optim, scheduler)

    trainer.pretrain_check = lambda *a, **kw: 0.0

    trainer.train_loop()

    latest = dirs['latest']
    assert (latest / "model.pt").exists()
    assert (latest / "optimizer.pt").exists()
    prog = json.loads((latest / "progress.json").read_text())
    assert 1 <= prog['step'] <= cfg.training.max_steps

    backups = sorted(dirs['backups'].glob("step_*.pt"))
    assert len(backups) <= 2

    ms = dirs['milestones']
    assert (ms / "step_2" / "model.pt").exists()
    assert (ms / "step_2" / "dist" / "model_inference.pt").exists()
    assert (ms / "step_4" / "model.pt").exists()


def test_interrupted_and_resume(test_env):
    tmp_path, shards, cfg, dirs = test_env
    ds = StreamingTextDataset(str(shards), context_length=128, shard_size=300)
    model = build_tiny_model(cfg)
    optim = get_optimizer(model, cfg)
    scheduler = get_scheduler(optim, cfg, total_steps=cfg.training.max_steps)
    trainer = Trainer(cfg, model, ds, optim, scheduler)

    trainer.pretrain_check = lambda *a, **kw: 0.0

    orig_forward = trainer._forward_backward
    call_count = {'count': 0}

    def limited_forward(inputs, labels):
        call_count['count'] += 1
        loss = orig_forward(inputs, labels)

        if call_count['count'] >= 2:
            raise KeyboardInterrupt()
        return loss
    trainer._forward_backward = limited_forward

    with pytest.raises(KeyboardInterrupt):
        trainer.train_loop()

    prog = json.loads((dirs['latest'] / "progress.json").read_text())
    assert prog['step'] == 3

    ds2 = StreamingTextDataset(
        str(shards), context_length=128, shard_size=300, resume_info=prog)
    trainer2 = Trainer(cfg, model, ds2, optim, scheduler)

    trainer2.pretrain_check = lambda *a, **kw: 0.0

    trainer2.train_loop()
    prog2 = json.loads((dirs['latest'] / "progress.json").read_text())
    assert prog2['step'] == cfg.training.max_steps

    assert (dirs['milestones'] / "step_6" /
            "model.pt").exists() or cfg.training.max_steps < 6


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
