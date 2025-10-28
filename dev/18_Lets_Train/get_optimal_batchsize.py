import os
import yaml
import json
import math
import time
import csv
import traceback
import datetime
import pytz
import argparse
import torch
from torch.utils.data import DataLoader

from architecture import DecoderOnlyTransformer
from data import MultiDirectoryStreamingTextDataset
from optimizer_config import create_optimizer, create_scheduler, get_fused_cross_entropy
from trainer import DecodeTransformerTrainer, TrainerConfig
from amd_gpu_setting import setup_complete_amd_environment, compile_model_amd

from dotenv import load_dotenv
load_dotenv()

data_folder_list = os.getenv("MYTLLM_DATA_LIST")
default_config = os.getenv("MYTLLM_CONFIG_BASE")


def load_config_from_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}

def merge_configs(base, override):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = merge_configs(base[k], v)
        else:
            base[k] = v
    return base

def fmt_time(seconds):
    return str(datetime.timedelta(seconds=seconds))


def generate_candidate_pairs(effective_bs):
    candidates = []
    max_power = int(math.log2(effective_bs))
    for p in range(max_power, 0, -1):
        mb = 2 ** p
        if effective_bs % mb == 0:
            accum = effective_bs // mb
            candidates.append((mb, accum))
    if effective_bs % 1 == 0:
        candidates.append((effective_bs, 1))
    return candidates


def try_combo(batch_size, accum_steps, final_config, directories, model_cfg, device, csv_writer):
    try:
        train_dataset = MultiDirectoryStreamingTextDataset(
            directories,
            context_length=model_cfg["context_length"]
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=0,  
            pin_memory=False,
            shuffle=False,
        )

        model = DecoderOnlyTransformer(
            vocab_size=model_cfg["vocab_size"],
            context_length=model_cfg["context_length"],
            model_dimension=model_cfg["dimension"],
            n_heads=model_cfg["n_heads"],
            Nx_blocks=model_cfg["num_layers"],
            ffn_hid_dim=model_cfg["hidden_dimension"],
            n_kv_heads=model_cfg["n_kv_heads"] if model_cfg["n_kv_heads"] != -1 else None,
            dropout=model_cfg["dropout"],
            tie_weights=model_cfg.get("tie_weights", True),
            use_checkpoint=model_cfg.get("use_checkpoint", False),
            checkpoint_ratio=model_cfg.get("checkpoint_ratio", 0.5),
        )

        if final_config['gpu']['name'] == 'AMD' and final_config['gpu']['compile']:
            precision = final_config['gpu'].get('precision', 'bf16')
            compile_approach = final_config['gpu'].get('compile_approach', 'full')
            amd_config = setup_complete_amd_environment(
                precision=precision,
                compile_approach=compile_approach
            )
            model, scaler, autocast_enabled = compile_model_amd(
                model,
                approach=compile_approach,
                precision=precision,
                set_precision=True
            )
            precision_cfg = {
                'scaler': scaler,
                'autocast_enabled': autocast_enabled,
                'precision': precision,
                'dtype': amd_config['dtype']
            }
        else:
            model = torch.compile(model, mode='reduce-overhead')
            precision_cfg = {
                'scaler': None,
                'autocast_enabled': False,
                'precision': 'fp32',
                'dtype': torch.float32
            }

        model.to(device)

        max_steps = final_config["max_steps"]
        optimizer = create_optimizer(model, lr=final_config["optimizer"]["lr"])
        scheduler = create_scheduler(optimizer, max_steps, warmup_steps=final_config["scheduler"]["warmup_steps"])
        loss_fn = get_fused_cross_entropy()

        # Trainer
        # trainer_cfg = TrainerConfig(**final_config["trainer"])
        # trainer_cfg.accum_steps = accum_steps
        # trainer = DecodeTransformerTrainer(
        #     model=model,
        #     optimizer=optimizer,
        #     loss_fn=loss_fn,
        #     train_loader=train_loader,
        #     val_loader=train_loader,
        #     device=device,
        #     scheduler=scheduler,
        #     precision_config=precision_cfg,
        #     trainer_config=trainer_cfg,
        #     verbose=False,
        # )

        warmup_steps = 16
        times = []
        accum = accum_steps
        micro = 0
        start_wall = time.time()

        for i, (x, y) in enumerate(train_loader):
            if i >= warmup_steps:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            preds = model(x)
            loss = loss_fn(preds, y) / accum
            loss.backward()

            micro += 1
            if micro % accum == 0:
                optimizer.step()
                scheduler.step()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                times.append(time.time() - start_wall)
                start_wall = time.time()

        if len(times) < 2:
            raise RuntimeError("Too few warmup steps to get stable timing")

        avg_time = sum(times[-2:]) / len(times[-2:])
        tokens_per_opt_step = batch_size * model_cfg["context_length"] * accum_steps
        time_for_1M_tokens = 1_000_000 * (avg_time / tokens_per_opt_step)

        print(f"[OK] BS={batch_size} ACC={accum_steps} → {time_for_1M_tokens:.2f}s/1M tok")
        csv_writer.writerow([batch_size, accum_steps, tokens_per_opt_step, avg_time, time_for_1M_tokens, "OK"])
        return True, time_for_1M_tokens

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[OOM] BS={batch_size} ACC={accum_steps}")
            csv_writer.writerow([batch_size, accum_steps, 0, 0, 0, "OOM"])
            return False, float('inf')
        else:
            print(f"[FAIL] BS={batch_size} ACC={accum_steps}: {e}")
            traceback.print_exc()
            csv_writer.writerow([batch_size, accum_steps, 0, 0, 0, f"FAIL:{e}"])
            return False, float('inf')
    except Exception as e:
        print(f"[FAIL] Unexpected for BS={batch_size}, ACC={accum_steps}: {e}")
        traceback.print_exc()
        csv_writer.writerow([batch_size, accum_steps, 0, 0, 0, f"FAIL:{e}"])
        return False, float('inf')


def main():
    final_config = load_config_from_yaml(default_config)
    model_cfg = final_config["model"]
    directories = json.loads(data_folder_list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_effective_bs = final_config.get("effective_batch_size", 1024)
    candidates = generate_candidate_pairs(target_effective_bs)

    print(f" Fast sweep for effective batch size = {target_effective_bs}")
    print(f"🔸 Candidate pairs (batch, accum): {candidates}")

    results_csv = "batch_sweep_BASE.csv"
    best_combo = None
    best_speed = float('inf')

    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_size", "accum_steps", "tokens_per_opt_step", "avg_time_per_opt_step", "time_per_1M_tokens", "status"])

        for bs, acc in candidates:
            ok, speed = try_combo(bs, acc, final_config, directories, model_cfg, device, writer)
            if ok and speed < best_speed:
                best_speed = speed
                best_combo = (bs, acc)

    print("\n==============================")
    print("✅ Sweep Completed")
    if best_combo:
        print(f"🏆 Best combo: batch_size={best_combo[0]}, accum_steps={best_combo[1]} (fastest 1M tokens={best_speed:.2f}s)")
    else:
        print("❌ No valid combination found")
    print(f"Results saved to {results_csv}")


if __name__ == "__main__":
    main()
