import os
import argparse
import yaml
import time
import json
import math
import pytz
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


from trainer import GeneralTrainer, TrainerConfig
from architecture import DecoderOnlyTransformer
from data import MultiDirectoryStreamingTextDataset, ValidationDatasetStreamer

from tokenizer import get_encoder, convert2hr, spl_tok_dict
from optimizer_config import create_optimizer, create_scheduler, get_fused_cross_entropy
from amd_gpu_setting import setup_complete_amd_environment, compile_model_amd

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.box import DOUBLE
from rich.box import HEAVY
console = Console()


from dotenv import load_dotenv
load_dotenv() 

data_folder = os.getenv("MYTLLM_DATA")
data_folder_list = os.getenv("MYTLLM_DATA_LIST")
default_config = os.getenv("MYTLLM_CONFIG")

def human(n):
    n = float(n)
    for u in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000.0:
            return f"{n:,.1f}{u}"
        n /= 1000.0
    return f"{n:.1f}P"

def fmt_time(total_seconds):
    total_seconds = float(total_seconds)
    neg = total_seconds < 0
    total_seconds = abs(total_seconds)
    d = int(total_seconds // 86400)
    h = int((total_seconds % 86400) // 3600)
    m = int((total_seconds % 3600) // 60)
    s = int(total_seconds % 60)
    return f"{'-' if neg else ''}{d}d {h:02d}h {m:02d}m {s:02d}s"

def fmt_dt_ist(dt):
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sept","Oct","Nov","Dec"]
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    ist = pytz.timezone("Asia/Kolkata")
    dt = dt.astimezone(ist)
    return f"{months[dt.month-1]} {dt.day:02d} {dt.year} {days[dt.weekday()]} {dt:%H:%M:%S}"



def get_time_str(time_in_secs):
    neg = False
    if time_in_secs <0:
        time_in_secs = abs(time_in_secs)
        neg = True
    days = int(time_in_secs // 86400)
    hours = int((time_in_secs % 86400) // 3600)
    minutes = int((time_in_secs % 3600) // 60)
    seconds = int(time_in_secs % 60)
    return f"{'-' if neg else ''} {days} Days {hours:2d} Hours {minutes:2d} Mins {seconds} Secs"



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

def overrides_to_dict(args):
    result = {}
    for key, value in vars(args).items():
        if value is None or key == "config_file":
            continue
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result

def parse_args():
    parser = argparse.ArgumentParser(description="Model training config overrides")
    parser.add_argument("--config_file", type=str, default=None, help="Optional additional YAML config file to override defaults")
    parser.add_argument('-t',"--train", action="store_true", help="starts training")
    parser.add_argument('-s',"--training.resume_train", action="store_false", help="set it for scratch")
    parser.add_argument('-g',"--generate", action="store_true", help="Toggle on to generate from model")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--optimizer.lr", type=float, help="Learning rate")
    parser.add_argument("--logging.log_dir", type=str, help="logging dir")
    return parser.parse_args()


def benchmark():
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    
    
    base_config = load_config_from_yaml(default_config)

    args = parse_args()

    if args.config_file:
        additional_config = load_config_from_yaml(args.config_file)
        base_config = merge_configs(base_config, additional_config)

    cli_override = overrides_to_dict(args)
    final_config = merge_configs(base_config, cli_override)
    
    m_cfg = final_config["model"]
    model_name = m_cfg.get("name", "model")
    context_len = m_cfg["context_length"]


    directory_list = json.loads(data_folder_list)
    train_dataset = MultiDirectoryStreamingTextDataset(
        directory_list, 
        context_length=m_cfg["context_length"],
        )
    val_dataset_streamer = ValidationDatasetStreamer(
        directory_list, 
        context_length=m_cfg["context_length"],
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DecoderOnlyTransformer(
        vocab_size=m_cfg["vocab_size"],
        context_length=m_cfg["context_length"],
        model_dimension=m_cfg["dimension"],
        n_heads=m_cfg["n_heads"],
        Nx_blocks=m_cfg["num_layers"],
        ffn_hid_dim=m_cfg["hidden_dimension"],
        n_kv_heads=(m_cfg["n_kv_heads"] if m_cfg["n_kv_heads"] != -1 else None),
        dropout=m_cfg["dropout"],
        tie_weights=m_cfg.get("tie_weights", True),
        use_checkpoint=m_cfg.get("use_checkpoint", False),
        checkpoint_ratio=m_cfg.get("checkpoint_ratio", 0.5),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    chinchilla_target_tokens   = 20 * n_params
    mahanthyalla_target_tokens = 39 * n_params

    sweep_cfg = final_config.get("sweep", {})
    batch_candidates = sweep_cfg.get("batch_sizes",  [4, 8, 16, 32])
    accum_candidates = sweep_cfg.get("accum_steps",  [1, 2, 4, 8])
    warmup_steps     = final_config["training"].get("train_time_warmup", 100)
    max_steps_plan   = final_config["training"]["max_steps"]
    log_dir          = final_config["logging"]["log_dir"]

    os.makedirs("benchmarks", exist_ok=True)
    def build_loaders(batch_size):
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=final_config["dataloader_settings"]["num_workers"],
            pin_memory=final_config["dataloader_settings"]["pin_memory"],
            persistent_workers=final_config["dataloader_settings"].get("persistent_workers", True),
            prefetch_factor=final_config["dataloader_settings"].get("prefetch_factor", 4),
        )
        val_loader = val_dataset_streamer.get_val_loader(
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=False
        )
        return train_loader, val_loader

    results = []
    total_trials = len(batch_candidates) * len(accum_candidates)
    trial_idx = 0

    for bs in batch_candidates:
        train_loader, val_loader = build_loaders(bs)
        micro_steps_per_epoch = len(train_loader)
        tokens_per_batch = bs * context_len
        for accum in accum_candidates:
            trial_idx += 1
            console.rule(f"[bold cyan]Trial {trial_idx}/{total_trials}[/bold cyan]  ─  batch={bs}, accum={accum}")

            optimizer = create_optimizer(model, lr=final_config["optimizer"]["lr"])
            scheduler = create_scheduler(optimizer, total_steps=max_steps_plan, warmup_steps=final_config["scheduler"]["warmup_steps"])
            loss_fn = get_fused_cross_entropy()

            trainer_cfg = TrainerConfig(
                accum_steps=accum,
                val_every_steps=final_config["training"]["eval_steps"],
                save_every_steps=final_config["training"]["save_steps"],
                keep_last=final_config["checkpoint"]["backups"]["keep_last"],
                bf16_autocast=True,
                async_ckpt_write=False,
            )

            trainer = GeneralTrainer(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                log_dir=log_dir,
                scheduler=scheduler,
                precision_config={"autocast_enabled": True, "dtype": torch.bfloat16},
                trainer_config=trainer_cfg,
                verbose= False,
            )

            status = "OK"
            avg_time_per_opt_step = float("inf")
            tokens_per_opt_step   = tokens_per_batch * accum
            opt_steps_per_epoch   = math.ceil(micro_steps_per_epoch / accum)
            tokens_per_epoch      = tokens_per_batch * micro_steps_per_epoch

            try:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                avg_time_per_opt_step = trainer.warmup_time_check_for_train(warmup_steps=warmup_steps)
                if avg_time_per_opt_step <= 0:
                    raise RuntimeError("Warmup returned non-positive time/step.")
            except torch.cuda.OutOfMemoryError:
                status = "OOM"
            except Exception as e:
                status = f"ERR:{type(e).__name__}"

            if status == "OK":
                time_epoch_sec    = opt_steps_per_epoch * avg_time_per_opt_step
                req_steps_chin    = math.ceil(chinchilla_target_tokens   / max(1, tokens_per_opt_step))
                req_steps_mahanth = math.ceil(mahanthyalla_target_tokens / max(1, tokens_per_opt_step))
                time_chin_sec     = req_steps_chin    * avg_time_per_opt_step
                time_mahanth_sec  = req_steps_mahanth * avg_time_per_opt_step
                time_plan_sec     = max_steps_plan    * avg_time_per_opt_step
            else:
                time_epoch_sec   = float("inf")
                time_chin_sec    = float("inf")
                time_mahanth_sec = float("inf")
                time_plan_sec    = float("inf")

            results.append({
                "batch_size": bs,
                "accum": accum,
                "tokens_per_batch": tokens_per_batch,
                "tokens_per_opt_step": tokens_per_opt_step,
                "micro_steps_per_epoch": micro_steps_per_epoch,
                "opt_steps_per_epoch": opt_steps_per_epoch,
                "tokens_per_epoch": tokens_per_epoch,
                "avg_time_per_opt_step_s": avg_time_per_opt_step,
                "time_epoch_s": time_epoch_sec,
                "time_chinchilla_s": time_chin_sec,
                "time_mahanthyalla_s": time_mahanth_sec,
                "time_plan_s": time_plan_sec,
                "status": status
            })

    ok_rows = [r for r in results if r["status"] == "OK"]
    if ok_rows:
        best = min(ok_rows, key=lambda r: r["time_chinchilla_s"])
    else:
        best = None

    tbl = Table(title=f"Hyper Sweep — {model_name}", box=DOUBLE)
    tbl.add_column("batch", justify="right")
    tbl.add_column("accum", justify="right")
    tbl.add_column("tok/opt", justify="right")
    tbl.add_column("t/opt (s)", justify="right")
    tbl.add_column("t_epoch", justify="right")
    tbl.add_column("t_chinchilla", justify="right")
    tbl.add_column("t_Mahanth", justify="right")
    tbl.add_column("status", justify="left")

    def fmt_s(sec):
        return "∞" if not math.isfinite(sec) else fmt_time(sec)

    for r in sorted(results, key=lambda z: (z["status"] != "OK", z["time_chinchilla_s"])):
        tbl.add_row(
            str(r["batch_size"]),
            str(r["accum"]),
            f"{r['tokens_per_opt_step']:,}",
            "∞" if not math.isfinite(r["avg_time_per_opt_step_s"]) else f"{r['avg_time_per_opt_step_s']:.4f}",
            fmt_s(r["time_epoch_s"]),
            fmt_s(r["time_chinchilla_s"]),
            fmt_s(r["time_mahanthyalla_s"]),
            ("[green]OK[/green]" if r["status"] == "OK" else f"[red]{r['status']}[/red]"),
        )

    console.print(tbl)

    if best:
        rec = (
            f"[bold green]BEST (by Chinchilla time):[/bold green] "
            f"batch={best['batch_size']}, accum={best['accum']} | "
            f"tok/opt={best['tokens_per_opt_step']:,} | "
            f"t/opt={best['avg_time_per_opt_step_s']:.4f}s | "
            f"t_chin={fmt_time(best['time_chinchilla_s'])}"
        )
        console.print(Panel(rec, border_style="green", title="Recommendation"))

    import csv
    csv_path = os.path.join("benchmarks", f"{model_name}_hyper.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "batch_size","accum","tokens_per_batch","tokens_per_opt_step",
            "micro_steps_per_epoch","opt_steps_per_epoch","tokens_per_epoch",
            "avg_time_per_opt_step_s","time_epoch_s","time_chinchilla_s","time_mahanthyalla_s",
            "time_plan_s","status"
        ])
        for r in results:
            writer.writerow([
                r["batch_size"], r["accum"], r["tokens_per_batch"], r["tokens_per_opt_step"],
                r["micro_steps_per_epoch"], r["opt_steps_per_epoch"], r["tokens_per_epoch"],
                r["avg_time_per_opt_step_s"], r["time_epoch_s"], r["time_chinchilla_s"], r["time_mahanthyalla_s"],
                r["time_plan_s"], r["status"]
            ])
    console.print(f"[bold cyan]Saved CSV:[/bold cyan] {csv_path}")

    try:
        import matplotlib.pyplot as plt

        ok_sorted = sorted([r for r in results if r["status"] == "OK"], key=lambda z: z["tokens_per_opt_step"])
        if ok_sorted:
            x = [r["tokens_per_opt_step"] for r in ok_sorted]

            y_epoch = [r["time_epoch_s"] for r in ok_sorted]
            plt.figure()
            plt.plot(x, y_epoch, marker="o")
            plt.xlabel("Tokens per optimizer step")
            plt.ylabel("Time per epoch (s)")
            plt.title(f"{model_name} — Epoch Time vs Tokens/Step")
            plt.grid(True)
            plt.tight_layout()
            out1 = os.path.join("benchmarks", f"{model_name}_epoch_time.png")
            plt.savefig(out1); plt.close()
            console.print(f"[bold cyan]Saved plot:[/bold cyan] {out1}")

            y_chin = [r["time_chinchilla_s"] for r in ok_sorted]
            plt.figure()
            plt.plot(x, y_chin, marker="o")
            plt.xlabel("Tokens per optimizer step")
            plt.ylabel("Time to Chinchilla (s)")
            plt.title(f"{model_name} — Chinchilla(20×) Time vs Tokens/Step")
            plt.grid(True)
            plt.tight_layout()
            out2 = os.path.join("benchmarks", f"{model_name}_chinchilla_time.png")
            plt.savefig(out2); plt.close()
            console.print(f"[bold cyan]Saved plot:[/bold cyan] {out2}")

            y_mah = [r["time_mahanthyalla_s"] for r in ok_sorted]
            plt.figure()
            plt.plot(x, y_mah, marker="o")
            plt.xlabel("Tokens per optimizer step")
            plt.ylabel("Time to Mahanthyalla (s)")
            plt.title(f"{model_name} — Mahanthyalla(39×) Time vs Tokens/Step")
            plt.grid(True)
            plt.tight_layout()
            out3 = os.path.join("benchmarks", f"{model_name}_mahanthyalla_time.png")
            plt.savefig(out3); plt.close()
            console.print(f"[bold cyan]Saved plot:[/bold cyan] {out3}")

        else:
            console.print("[yellow]No successful trials to plot.[/yellow]")
    except Exception as e:
        console.print(f"[red]Plotting failed:[/red] {e}")






if __name__ == "__main__":
    start = datetime.datetime.now()
    benchmark()
    console.print('[bold green]✅ Done! Script Completed Successfuly[/bold green]')
    print(f'⌚ Script Time: {datetime.datetime.now() - start}')
    