
# MYT-LLM

**MYT-LLM** is a configurable, production-ready training ecosystem for decoder-only transformers, optimized for single AMD MI300X GPUs and large streaming datasets.

## Features

- **Config-driven**: YAML configs for full-scale, small, and tiny models  
- **Streaming Dataset**: Dynamic shard loading with resume support  
- **Model Architecture**: Pre-norm transformer, SwiGLU, RMSNorm, FlashAttention  
- **Training Loop**: Pre-train checks, tqdm logging, mixed precision ready  
- **Checkpointing**: Latest, backups, and milestones (with dist exports)  
- **Resume**: Automatic recovery from interruptions  
- **Logging & Visualization**: File + console logging, TensorBoard, Matplotlib plots  
- **Sampling**: Top-k/top-p with temperature  
- **Scripts**: `main.py`, `dry_run.py`, `generate_samples.py`, `visualize_metrics.py`  
- **Tests**: Comprehensive unit tests for all modules

## Directory Structure

```
myt-llm/
├── configs/          # YAML configs
├── data/             # Streaming dataset & collate
├── model/            # Transformer architecture
├── training/         # Trainer, callbacks, scheduler, resume
├── utils/            # Logging, metrics, viz, sampling, checkpointing
├── scripts/          # CLI entrypoints
├── tests/            # PyTest unit tests
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Quick Start

1. Install dependencies:  
   ```
   pip install -r requirements.txt
   ```
2. Prepare shards under `data/shards/`.  
3. Edit `configs/default.yaml` and `configs/train_*.yaml`.  
4. Launch training:  
   ```
   python scripts/main.py --config configs/train_mytlm.yaml
   ```
5. Monitor logs in `checkpoints/logs/` and TensorBoard in `runs/`.  
6. Generate samples:  
   ```
   python scripts/generate_samples.py --config configs/train_mytlm.yaml \
     --checkpoint checkpoints/latest/model.pt --prompt "42"
   ```
7. Visualize metrics:  
   ```
   python scripts/visualize_metrics.py --metrics checkpoints/milestones/step_10000/metrics.json
   ```

## Testing

Run all tests with:
```
# pytest -q
python -m pytest tests -q
```

### individual tests
#### DataSet Loader : Testing

To run the tests and log results:

```bash
python -m tests.test_dataset -q
```

#### model testing 

To run tests and log results:

```bash
python -m tests.test_model -q
```

#### utils testing 
To run the tests:

```bash
python -m tests.test_utils -q
```