import os
import shutil
import tempfile
import json
import subprocess
import torch
import pytest
import yaml
from pathlib import Path

from model.architecture import DecoderOnlyTransformer
from data.streaming_dataset import StreamingTextDataset


def setup_test_env(tmp_path, vocab_size=1024):

    base = tmp_path
    shards = base / "data" / "shards"
    part = shards / "part-000"
    part.mkdir(parents=True)

    for i in range(2):
        tokens = torch.arange(vocab_size, dtype=torch.long)
        torch.save(tokens, part / f"shard_{i:05d}.pt")

    cfg = {
        "device": "cpu",
        "batch_size": 2,

        "dataset": {"shard_folder": str(shards), "context_length": 128},
        "model": {
            "vocab_size": vocab_size,
            "context_length": 128,
            "dimension": 32,
            "n_heads": 4,
            "num_layers": 1,
            "hidden_dimension": 64,
            "non_linearity": "gelu",
            "dropout": 0.0,
            "tie_weights": True,
        },
        "dataloader_settings": {"num_workers": 0, "pin_memory": False},
        "optimizer": {"lr": 1e-3, "weight_decay": 0.0},
        "scheduler": {"warmup_steps": 1},
        "logging": {
            "interval_steps": 1,
            "log_files_dir": str(base / "logs"),
            "train_log_file": str(base / "logs/train.log"),
            "samples_log_file": str(base / "logs/gen_samples.log"),
        },
        "checkpoint": {
            "dir": str(base / "checkpoints" / "latest"),
            "backups": {"dir": str(base / "checkpoints" / "backups"), "keep_last": 1},
        },
        "milestones": {
            "dir": str(base / "checkpoints" / "milestones"),
            "interval_steps": 1,
            "dist_subfolder": "dist",
        },
        "resume": {"enabled": False},
        "training": {"use_checkpoint": False, "max_steps": 2},
    }

    cfg_path = base / "test_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    return base, cfg_path


@pytest.mark.parametrize(
    "script, args, expect_file",
    [
        ("main.py", "--config", "checkpoints/latest/model.pt"),
        ("dry_run.py", "--config", None),
        ("generate_samples.py", "--config", None),
        ("visualize_metrics.py", "--metrics", None),
    ],
)
def test_scripts(tmp_path, script, args, expect_file):
    base, cfg_path = setup_test_env(tmp_path, vocab_size=1024)

    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / script

    if script == "visualize_metrics.py":

        metrics = {"steps": [0, 1], "loss": [
            1.0, 0.5], "perplexity": [2.7, 1.6]}
        ms_dir = base / "checkpoints" / "milestones" / "step_1"
        ms_dir.mkdir(parents=True)
        metrics_file = ms_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f)
        cmd = [
            "python",
            str(script_path),
            args,
            str(metrics_file),
            "--output_dir",
            str(base / "plots"),
        ]

    elif script == "generate_samples.py":

        ckpt_dir = base / "checkpoints" / "latest"
        ckpt_dir.mkdir(parents=True)

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        m_cfg = cfg["model"]
        dummy_model = DecoderOnlyTransformer(
            vocab_size=m_cfg["vocab_size"],
            context_length=m_cfg["context_length"],
            model_dimension=m_cfg["dimension"],
            n_heads=m_cfg["n_heads"],
            Nx_blocks=m_cfg["num_layers"],
            ffn_hid_dim=m_cfg["hidden_dimension"],
            non_linearity=m_cfg["non_linearity"],
            dropout=m_cfg["dropout"],
        )

        torch.save(dummy_model.state_dict(), ckpt_dir / "model.pt")
        cmd = [
            "python",
            str(script_path),
            "--config",
            str(cfg_path),
            "--checkpoint",
            str(ckpt_dir / "model.pt"),
            "--prompt",
            "1 2 3",
        ]

    else:
        cmd = ["python", str(script_path), "--config", str(cfg_path)]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    result = subprocess.run(
        cmd, cwd=tmp_path, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr

    if expect_file:
        assert (tmp_path / expect_file).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
