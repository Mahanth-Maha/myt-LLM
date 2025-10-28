import os
import yaml
import json
import math
import time
import csv
import traceback
import torch
from torch.utils.data import DataLoader

from architecture import DecoderOnlyTransformer
from data import MultiDirectoryStreamingTextDataset
from optimizer_config import create_optimizer, create_scheduler, get_fused_cross_entropy
from trainer import DecodeTransformerTrainer, TrainerConfig
from amd_gpu_setting import setup_complete_amd_environment, compile_model_amd

from dotenv import load_dotenv
load_dotenv()

# ========== Load ENV ==========
data_folder_list = os.getenv("MYTLLM_DATA_LIST")
# model_name_config = 'SMALL'
# MIN_POWERS = 5 # working bs 
# MAX_POWERS = 12
# model_name_config = 'TINY'
# MIN_POWERS = 6 # working bs 
# MAX_POWERS = 15
# model_name_config = 'NANO'
# MIN_POWERS = 9 # working bs 
# MAX_POWERS = 15
# model_name_config = 'TINY_DEEP'
# MIN_POWERS = 5 # working bs 
# MAX_POWERS = 15
model_name_config = 'SMALL_DEEP'
MIN_POWERS = 5 # working bs 
MAX_POWERS = 15

default_config = os.getenv(f"MYTLLM_CONFIG_{model_name_config}")

def load_config_from_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def warmup_timing_for_batch(model, train_loader, optimizer, scheduler, loss_fn, device, warmup_steps = 10):
    model.train()
    times = []
    start = time.time()
    micro = 0

    for i, (x, y) in enumerate(train_loader):
        if i >= warmup_steps:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        preds = model(x)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
        times.append(end - start)
        start = time.time()

    if len(times) == 0:
        return float("inf")
    return sum(times) / len(times)


def try_batch_size(batch_size, final_config, directories, model_cfg, device, csv_writer):
    try:
        dataset = MultiDirectoryStreamingTextDataset(
            directories,
            context_length=model_cfg["context_length"]
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            shuffle=False
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
        else:
            model = torch.compile(model, mode='reduce-overhead')
        
        model.to(device)

        max_steps = final_config["max_steps"]
        optimizer = create_optimizer(model, lr=final_config["optimizer"]["lr"])
        scheduler = create_scheduler(optimizer, max_steps, warmup_steps=final_config["scheduler"]["warmup_steps"])
        loss_fn = get_fused_cross_entropy()


        avg_time = warmup_timing_for_batch(model, loader, optimizer, scheduler, loss_fn, device)
        if avg_time == float("inf"):
            raise RuntimeError("Warmup failed or too few steps")

        tokens_per_step = batch_size * model_cfg["context_length"]
        time_for_1M = 1_000_000 * (avg_time / tokens_per_step)
        csv_writer.writerow([batch_size, avg_time, time_for_1M, "OK"])
        print(f"[OK] BS={batch_size}: {time_for_1M:.2f}s / 1M tokens")
        return True, time_for_1M

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            csv_writer.writerow([batch_size, 0, 0, "OOM"])
            print(f"[OOM] BS={batch_size}")
            return False, float("inf")
        else:
            csv_writer.writerow([batch_size, 0, 0, f"FAIL:{e}"])
            print(f"[FAIL] BS={batch_size}: {e}")
            traceback.print_exc()
            return False, float("inf")
    except Exception as e:
        csv_writer.writerow([batch_size, 0, 0, f"FAIL:{e}"])
        print(f"[FAIL] BS={batch_size}: {e}")
        traceback.print_exc()
        return False, float("inf")


def find_largest_batch(final_config, directories, model_cfg, device):
    results_csv = f"batch_sweep_binary_{model_name_config}.csv"
    best_bs = None
    best_speed = float("inf")

    # max_bs_test =   # upper cap to avoid crazy OOMs
    power_candidates = [2**i for i in range(MIN_POWERS, MAX_POWERS)]  # 32..16384
    # power_candidates = [bs for bs in power_candidates if bs <= max_bs_test]

    last_good = None
    first_fail = None

    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_size", "avg_time_per_step", "time_per_1M_tokens", "status"])

        for bs in power_candidates:
            ok, speed = try_batch_size(bs, final_config, directories, model_cfg, device, writer)
            if ok:
                last_good = bs
                if speed < best_speed:
                    best_bs, best_speed = bs, speed
            else:
                first_fail = bs
                break

        if last_good and first_fail:
            low, high = last_good, first_fail
            while high - low > 4:
                mid = (low + high) // 2
                ok, speed = try_batch_size(mid, final_config, directories, model_cfg, device, writer)
                if ok:
                    low = mid
                    if speed < best_speed:
                        best_bs, best_speed = mid, speed
                else:
                    high = mid

    return best_bs, best_speed


def main():
    print(" Batch Size Finder (accum_steps=1)")
    final_config = load_config_from_yaml(default_config)
    model_cfg = final_config["model"]
    directories = json.loads(data_folder_list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_bs, best_speed = find_largest_batch(final_config, directories, model_cfg, device)

    print("\n==============================")
    if best_bs:
        print(f" Model: {model_name_config}")
        print(f" Recommended batch size: {best_bs}")
        print(f" Estimated speed: {best_speed:.2f}s / 1M tokens")
    else:
        print(" No valid batch size found.")


if __name__ == "__main__":
    main()
