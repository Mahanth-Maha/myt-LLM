# MYT-LLM Quick Start

This guide shows how to install dependencies, configure, and run a training job on MYT-LLM.

## 1. Install Dependencies

```bash
git clone https://github.com/your_org/myt-llm.git
cd myt-llm
pip install -r requirements.txt
```

## 2. Prepare Data Shards

Place tokenized `.pt` shard files under `data/shards/`, preserving any nested folders.  
Each shard file must be named `shard_<5digit>.pt`. Example:

```bash
data/shards/
└── part-000/
    ├── shard_00000.pt
    └── shard_00001.pt
```

## 3. Configure Training

Edit `configs/default.yaml` for global settings (logging, checkpoint paths).  
Select a model config:

- `configs/train_mytlm.yaml` (1 B parameters)  
- `configs/train_mytlm-small.yaml` (258 M parameters)  
- `configs/train_mytlm-nano.yaml` (73 M parameters)  

You can override any setting via CLI:
```bash
# Example overriding batch size
python -m scripts.main \
  --config configs/train_mytlm.yaml \
  training.batch_size=4
```

## 4. Run a Dry-Run

Validate data loading and estimate iteration time:
```bash
python -m scripts.dry_run --config configs/train_mytlm-nano.yaml
```

This runs 10 warm-up iterations and reports average time per batch.

## 5. Start Training

```bash
python -m scripts.main --config configs/train_mytlm-nano.yaml
```

- **Checkpoints** are saved to `checkpoints/latest/`.  
- **Backups** (two most recent) in `checkpoints/backups/`.  
- **Milestones** every 10 000 steps under `checkpoints/milestones/step_<N>/`.  
- **Logs** under `checkpoints/logs/`.  
- **TensorBoard** events in `runs/`.

## 6. Resume Training

If interrupted, rerun the same command; training will resume from the last checkpoint.

## 7. Generate Samples

```bash
python -m scripts.generate_samples \
  --config configs/train_mytlm-nano.yaml \
  --checkpoint checkpoints/latest/model.pt \
  --prompt "101 202 303" \
  --max_tokens 50 \
  --temp 0.8 \
  --top_k 50 \
  --top_p 0.9
```

Generated token IDs and prompt are logged to `checkpoints/logs/gen_samples.log`.

## 8. Visualize Metrics

```bash
python -m scripts.visualize_metrics \
  --metrics checkpoints/milestones/step_100/metrics.json \
  --output_dir plots/
```

Loss and perplexity curves are saved under `plots/`.

---

For full documentation, see `README.md`.









# Quick Check - Tiny Model

To verify the end-to-end flow on the smallest model (mytlm-tiny), run:

1. Dry-run (10 warm-up iterations):  
```bash
python -m scripts.dry_run --config configs/train_mytlm-tiny.yaml
```

2. Quick train for 20 steps:  
```bash
python -m scripts.main --config configs/train_mytlm-tiny.yaml
```

3. Generate a few sample tokens:  
```bash
python -m scripts.generate_samples \
  --config configs/train_mytlm-tiny.yaml \
  --checkpoint checkpoints/latest/model.pt \
  --prompt "1 2 3" \
  --max_tokens 10 \
  --temp 1.0 \
  --top_k 5
```

4. Visualize metrics from the first milestone (step 10):  
```bash
python -m scripts.visualize_metrics \
  --metrics checkpoints/milestones/step_10/metrics.json \
  --output_dir plots/
```