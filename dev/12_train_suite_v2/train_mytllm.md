
# Training MYT-LLM (1 B Parameters)

This guide details how to train the full-scale MYT-LLM model (≈1 B parameters) on a single AMD MI300X GPU.

## Model Configuration

File: `configs/train_mytlm.yaml`

```
model:
  name: mytlm
  vocab_size: 100352
  context_length: 1024
  dimension: 1024
  hidden_dimension: 6144
  n_heads: 16
  num_layers: 39
  non_linearity: swiglu
  dropout: 0.1
  tie_weights: true

dataset:
  shard_folder: ./data/shards
  context_length: 1024

training:
  max_steps: 500000
  batch_size: 4
  gradient_accumulation_steps: 4
  use_checkpoint: true

optimizer:
  name: AdamW
  lr: 2e-4
  weight_decay: 0.1

scheduler:
  type: cosine
  warmup_steps: 10000

checkpoint:
  dir: ./checkpoints/latest
  backups:
    dir: ./checkpoints/backups
    keep_last: 2

milestones:
  dir: ./checkpoints/milestones
  interval_steps: 10000
  dist_subfolder: dist

logging:
  interval_steps: 100
  log_files_dir: ./checkpoints/logs
  train_log_file: ./checkpoints/logs/train.log
  tensorboard_log_dir: ./runs

resume:
  enabled: true
  progress_file: ./checkpoints/latest/progress.json
```

## Hardware & Precision

- **GPU**: AMD MI300X (192 GB HBM3)  
- **Precision**: BF16 mixed precision (configured in `configs/default.yaml`)  
- **Performance Tips**:  
  - Enable `cudnn.benchmark` and TF32 if using CUDA.  
  - Use gradient checkpointing to fit long sequences.  

## Training Command

```
python scripts/main.py --config configs/train_mytlm.yaml
```

### Monitoring

- **TQDM** progress bar prints steps and ETA.  
- **Logs**:  
  - `checkpoints/logs/train.log` for detailed training info.  
  - `runs/` folder for TensorBoard:  
    ```
    tensorboard --logdir runs/
    ```  
- **Checkpoints**:  
  - `checkpoints/latest/` latest state.  
  - `checkpoints/backups/` two most recent backups.  
  - `checkpoints/milestones/step_<N>/` model, optimizer, scheduler, `metrics.json`, and `dist/model_inference.pt`.

## Resume & Fault Tolerance

If training stops unexpectedly:

```
python scripts/main.py --config configs/train_mytlm.yaml
```

The trainer automatically loads `checkpoints/latest/progress.json` and skips processed tokens.

## Sample Generation

After milestones, generate inference samples:

```
python scripts/generate_samples.py \
  --config configs/train_mytlm.yaml \
  --checkpoint checkpoints/milestones/step_10000/dist/model_inference.pt \
  --prompt "123 456" \
  --max_tokens 100
```

## Visualization

Plot metrics from any milestone:

```
python scripts/visualize_metrics.py \
  --metrics checkpoints/milestones/step_10000/metrics.json
```

---

This setup will pretrain MYT-LLM at scale, with robust checkpointing, logging, and resume support.