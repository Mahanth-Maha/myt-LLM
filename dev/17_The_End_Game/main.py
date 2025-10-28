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

from tokenizer import ENDOFTEXT, get_encoder, convert2hr, spl_tok_dict
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
            return f"{n:,.1f} {u}"
        n /= 1000.0
    return f"{n:.1f} P"

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

def main2():
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    base_config = load_config_from_yaml(default_config)
    
    args = parse_args()
    
    if args.config_file:
        user_config = load_config_from_yaml(args.config_file)
        base_config = merge_configs(base_config, user_config)
    
    cli_overrides = overrides_to_dict(args)
    final_config = merge_configs(base_config, cli_overrides)

    m_cfg = final_config["model"]
    batch_size = final_config["batch_size"]
    grad_accum = final_config["trainer"]['accum_steps']
    context_length = m_cfg["context_length"]

    directories = json.loads(data_folder_list)
    train_dataset = MultiDirectoryStreamingTextDataset(
        directories, 
        context_length=context_length
    )
    val_dataset_streamer = ValidationDatasetStreamer(
        directories, 
        context_length=context_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=final_config["dataloader_settings"]["num_workers"],
        pin_memory=final_config["dataloader_settings"]["pin_memory"],
        prefetch_factor=final_config["dataloader_settings"]["prefetch_factor"],
        persistent_workers=final_config["dataloader_settings"]["persistent_workers"],
        shuffle=False,
    )
    val_loader = val_dataset_streamer.get_val_loader(
        batch_size=batch_size,
        num_workers=final_config["dataloader_settings"]["val_num_workers"],
        pin_memory=final_config["dataloader_settings"]["val_pin_memory"],
        shuffle=False
    )
    
    if final_config["device"] =='cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    model = DecoderOnlyTransformer(
        vocab_size=m_cfg["vocab_size"],
        context_length=m_cfg["context_length"],
        model_dimension=m_cfg["dimension"],
        n_heads=m_cfg["n_heads"],
        Nx_blocks=m_cfg["num_layers"],
        ffn_hid_dim=m_cfg["hidden_dimension"],
        n_kv_heads=m_cfg["n_kv_heads"] if m_cfg["n_kv_heads"] != -1 else None,
        dropout=m_cfg["dropout"],
        tie_weights=m_cfg.get("tie_weights", True),
        use_checkpoint=final_config["model"].get("use_checkpoint", False),
        checkpoint_ratio=final_config["model"].get("checkpoint_ratio", 0.5),
    )

    model_params = sum(p.numel() for p in model.parameters())

    console.print(Panel.fit(
        f"[bold cyan]📌 Model:[/bold cyan] {m_cfg['name']}\n"
        f"[bold]Parameters:[/bold] {model_params:,} ({convert2hr(model_params)})\n"
        f"[bold]Context Length:[/bold] {context_length} | [bold]Batch Size:[/bold] {batch_size}\n"
        f"[bold]Device:[/bold] {device}",
        title="🧠 Model Summary",
        border_style="cyan"
    ))

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
        
        training_precision = {
            'scaler': scaler,
            'autocast_enabled': autocast_enabled,
            'precision': precision,
            'dtype': amd_config['dtype']
        }
        
    else:
        print('Compiling with standard PyTorch...')
        model = torch.compile(model, mode='max-autotune')
        training_precision = {
            'scaler': None,
            'autocast_enabled': False,
            'precision': 'fp32',
            'dtype': torch.float32
        }
    
    print('✅ Compiling Done')
    
    
    max_steps = final_config["max_steps"]
    optimizer = create_optimizer(model, lr=final_config["optimizer"]["lr"])
    scheduler = create_scheduler(optimizer, max_steps, warmup_steps=final_config["scheduler"]["warmup_steps"])
    loss_fn = get_fused_cross_entropy()

    trainer_config = TrainerConfig(**final_config["trainer"])
    trainer = GeneralTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        log_dir=final_config["logging"]["log_dir"],
        scheduler=scheduler,
        metric_function=None,
        precision_config=training_precision,
        trainer_config=trainer_config,
        verbose=True,
    )

    # avg_time = trainer.warmup_time_check_for_train(warmup_steps=final_config['trainer']['train_time_warmup'])
    # micro_steps_per_epoch = len(train_loader)
    # opt_steps_per_epoch = math.ceil(micro_steps_per_epoch / grad_accum)
    # tokens_per_micro = batch_size * context_length
    # tokens_per_step = tokens_per_micro * grad_accum
    # tokens_per_epoch = tokens_per_micro * micro_steps_per_epoch
    # total_train_tokens = tokens_per_step * max_steps

    # n_params = model_params
    # chinchilla_target = 20 * n_params
    # mahanthyalla_target = 39 * n_params

    # eta = datetime.datetime.now() + datetime.timedelta(seconds=max_steps * avg_time)
    # eta_ist = eta.astimezone(pytz.timezone("Asia/Kolkata"))

    # # table1_1 = Table(title="📊 Dataset & Model", box=DOUBLE)
    # table1_1 = Table(title=None, box=DOUBLE)
    # table1_1.add_column("Metric", style="cyan")
    # table1_1.add_column("Value", justify="right", style="bold")
    # table1_1.add_row("Micro Steps / Epoch", f"{micro_steps_per_epoch:,}")
    # table1_1.add_row("Optimizer Steps / Epoch", f"{opt_steps_per_epoch:,}")
    # table1_1.add_row("Tokens / Microbatch", f"{tokens_per_micro:,}")
    # table1_1.add_row("Tokens / Optimizer Step", f"{tokens_per_step:,}")
    
    # # table1_2 = Table(title="📊 Tokens", box=DOUBLE)
    # table1_2 = Table(title=None, box=DOUBLE)
    # table1_2.add_column("Metric", style="cyan")
    # table1_2.add_column("Tokens", justify="right", style="bold")
    # table1_2.add_row("Tokens / Epoch", convert2hr(tokens_per_epoch))
    # table1_2.add_row("Total Train Tokens", convert2hr(total_train_tokens))
    # table1_2.add_row("Chinchilla Target", convert2hr(chinchilla_target))
    # table1_2.add_row("Mahanth Yalla Target (39×)", convert2hr(mahanthyalla_target))

    # table2 = Table(title="⏱ ETA & Runtime", box=DOUBLE)
    # table2.add_column("Metric", style="green")
    # table2.add_column("Tokens", justify="right", style="bold")
    # table2.add_row("Time / Step", f"{avg_time:.4f} s")
    # table2.add_row("Total Steps", f"{max_steps:,}")
    # table2.add_row("Est. Total Time", get_time_str(max_steps * avg_time))
    # table2.add_row("ETA (IST)", eta_ist.strftime('%Y-%m-%d %H:%M:%S'))

    # console.print(Panel(table1_1, title="[bold yellow]Dataset & Model[/bold yellow]", border_style="yellow"))
    # console.print(Panel(table1_2, title="[bold cyan]Tokens Stats[/bold cyan]", border_style="cyan"))
    # console.print(Panel(table2, title="[bold green]Time Estimates[/bold green]", border_style="green"))



    # Warmup measures **optimizer step** time in your upgraded trainer
    avg_time_per_opt_step = trainer.warmup_time_check_for_train(warmup_steps=final_config['trainer']['train_time_warmup'])

    # Loader & token accounting
    micro_steps_per_epoch = len(train_loader)                        # dataloader iterations per epoch (aka "batches")
    opt_steps_per_epoch   = math.ceil(micro_steps_per_epoch / grad_accum)

    tokens_per_batch      = batch_size * context_length               # tokens in one dataloader batch (micro-step)
    tokens_per_opt_step   = tokens_per_batch * grad_accum           # tokens accumulated before one optimizer step
    tokens_per_epoch      = tokens_per_batch * micro_steps_per_epoch

    # Totals for current run
    total_steps           = final_config['max_steps']   # planned optimizer steps
    total_train_tokens    = tokens_per_opt_step * total_steps

    # Model scale rules
    n_params              = model_params if 'model_params' in locals() else sum(p.numel() for p in model.parameters())
    chinchilla_target     = 20 * n_params                           # “Chinchilla(20×)”
    mahanthyalla_target   = 39 * n_params                           # “Mahanthyalla(39×)”

    def shortfall_or_surplus(target_tokens, have_tokens):
        delta = have_tokens - target_tokens
        if delta >= 0:
            return f"[green]surplus +{human(delta)} tks[/green]"
        else:
            return f"[red]shortfall -{human(-delta)} tks[/red]"

    # Recommended settings to hit targets (keep batch & context fixed; adjust steps or accum)
    def recommend_settings(target_tokens):
        # minimum optimizer steps needed if we keep current tokens_per_opt_step
        req_steps = math.ceil(target_tokens / tokens_per_opt_step)
        # if we want to keep max_steps fixed, compute required accum steps
        req_accum_if_fix_steps = math.ceil(max(1, target_tokens / (tokens_per_batch * total_steps)))
        return req_steps, req_accum_if_fix_steps

    chin_req_steps, chin_req_accum = recommend_settings(chinchilla_target)
    mah_req_steps,  mah_req_accum  = recommend_settings(mahanthyalla_target)

    time_per_micro_step   = avg_time_per_opt_step / max(1, grad_accum)
    time_per_opt_step     = avg_time_per_opt_step
    time_for_1M_tokens    = 1_000_000 * (time_per_opt_step / tokens_per_opt_step)
    time_one_epoch        = opt_steps_per_epoch * time_per_opt_step
    time_total_run        = total_steps * time_per_opt_step
    time_chinchilla       = chin_req_steps * time_per_opt_step
    time_mahanthyalla     = mah_req_steps  * time_per_opt_step

    current_fixed = datetime.datetime.now(pytz.timezone("Asia/Kolkata"))
    eta_run       = current_fixed + datetime.timedelta(seconds=time_total_run)
    eta_chin      = current_fixed + datetime.timedelta(seconds=time_chinchilla)
    eta_mah       = current_fixed + datetime.timedelta(seconds=time_mahanthyalla)
    eta_one_epoch = current_fixed + datetime.timedelta(seconds=time_one_epoch)

    box0 = f"""
    [bold cyan]MODEL OVERVIEW[/bold cyan]

    • Model Name               : [bold]{m_cfg['name']}[/bold]
    • Model parameters count   : [bold]{convert2hr(model_params)}[/bold]         ([dim]{model_params:,}[/dim])
    • Settings
        vocab_size        :[bold]{m_cfg['vocab_size']}[/bold]
        context_length    :[bold]{m_cfg['context_length']}[/bold]
        model_dimension   :[bold]{m_cfg['dimension']}[/bold]
        n_heads           :[bold]{m_cfg['n_heads']}[/bold]
        Nx_blocks         :[bold]{m_cfg['num_layers']}[/bold]
        ffn_hid_dim       :[bold]{m_cfg['hidden_dimension']}[/bold]
        n_kv_heads        :[bold]{m_cfg['n_kv_heads'] if m_cfg['n_kv_heads'] != -1 else None}[/bold]
        dropout           :[bold]{m_cfg['dropout']}[/bold]
    """
    box1 = f"""
    [bold cyan]DATASET OVERVIEW[/bold cyan]

    • Tokens in full dataset (per epoch)      : [bold]{convert2hr(tokens_per_epoch)}[/bold] tokens ([dim]{tokens_per_epoch:,}[/dim])
    • Dataloader iterations per epoch         : [bold]{micro_steps_per_epoch:,}[/bold] iters
    • With your current accumulation:
        - Batch Size                          : [bold]{batch_size:,}[/bold]
        - Accumulation Steps                  : [bold]{final_config['trainer']['accum_steps']:,}[/bold]
        - Optimizer steps per epoch           : [bold]{opt_steps_per_epoch:,}[/bold]

    [dim]Legend:[/dim]
        • [bold]Iteration[/bold] = one dataloader batch → “micro-step”
        • [bold]Optimizer step[/bold] = after accumulating [bold]{grad_accum}[/bold] iterations, we step the optimizer
    """

    box2 = f"""
    [bold magenta]TOKENS & BATCHING[/bold magenta]

    • Given batch size                         : [bold]{batch_size}[/bold]
    • Context length (tokens / sample)         : [bold]{context_length}[/bold]
    • Gradient accumulation steps              : [bold]{grad_accum}[/bold]
    • Effective batch size (BATCH_SIZE)        : [bold]{grad_accum * batch_size}[/bold]

    • Tokens per iteration (one dataloader batch)     : [bold]{tokens_per_batch:,}[/bold]
    • Tokens per optimizer step (after accumulation)  : [bold]{tokens_per_opt_step:,}[/bold]
    • Planned steps in this run                       : [bold]{total_steps:,}[/bold]
    • [bold]Total train tokens in this run[/bold]                     : [bold]{convert2hr(total_train_tokens)}[/bold]  ([dim]{total_train_tokens:,}[/dim])

    [bold magenta]Targets[/bold magenta]
    • Model parameters                  : [bold]{convert2hr(n_params)}[/bold]  ([dim]{n_params:,}[/dim])
    • Training tokens                   : [bold]{convert2hr(total_train_tokens)}[/bold]  ([dim]{total_train_tokens:,}[/dim])
    • Chinchilla target  (20× params)   : [bold]{convert2hr(chinchilla_target)}[/bold](needed)  → {shortfall_or_surplus(chinchilla_target, total_train_tokens)}
    • Mahanthyalla target (39× params)  : [bold]{convert2hr(mahanthyalla_target)}[/bold](needed)  → {shortfall_or_surplus(mahanthyalla_target, total_train_tokens)}

    [bold magenta]To hit targets (keeping batch={batch_size}, context={context_length})[/bold magenta]
    • Chinchilla:
        - Required optimizer steps                    : [bold]{chin_req_steps:,}[/bold]
        - Or keep steps={total_steps:,} → set accum   : [bold]{chin_req_accum}[/bold]
    • Mahanth Yalla:
        - Required optimizer steps                    : [bold]{mah_req_steps:,}[/bold]
        - Or keep steps={total_steps:,} → set accum   : [bold]{mah_req_accum}[/bold]
    """

    box3 = f"""
    [bold green]TIME & ETAs[/bold green]

    [bold]Per-unit times[/bold]
    • Time per iteration (one dataloader batch)\t: [bold]{time_per_micro_step:.4f} s[/bold]
    • Time per optimizer step (after {grad_accum} iters)\t: [bold]{time_per_opt_step:.4f} s[/bold]
    • Time for 1M tokens (at current mix)\t: [bold]{fmt_time(time_for_1M_tokens)}[/bold]

    [bold]Larger spans[/bold]
    • One full epoch time\t\t: [bold]{fmt_time(time_one_epoch)}[/bold]
    • Planned run time (steps={total_steps:,})\t\t: [bold]{fmt_time(time_total_run)}[/bold]
    • If steps set to Chinchilla\t\t: [bold]{fmt_time(time_chinchilla)}[/bold]
    • If steps set to Mahanthyalla\t\t: [bold]{fmt_time(time_mahanthyalla)}[/bold]

    [bold]Calendar[/bold]
    • Current time                  : [bold]{fmt_dt_ist(current_fixed)}[/bold]
    • ETA for planned run           : [bold]{fmt_dt_ist(eta_run)}[/bold]
    • ETA if Chinchilla steps       : [bold]{fmt_dt_ist(eta_chin)}[/bold]
    • ETA if Mahanthyalla steps     : [bold]{fmt_dt_ist(eta_mah)}[/bold]
    • ETA after one full epoch      : [red]{fmt_dt_ist(eta_one_epoch)}[/red]
    """
    print(f'\n\n')
    console.print(Panel.fit(box0, title="🧠 Model", border_style="yellow", box=HEAVY))
    console.print(Panel.fit(box1, title="📦 Dataset", border_style="cyan", box=HEAVY))
    console.print(Panel.fit(box2, title="🧮 Tokens & Targets", border_style="magenta", box=HEAVY))
    console.print(Panel.fit(box3, title="⏱️  Time Estimates", border_style="green", box=HEAVY))
    print(f'\n\n')


    if args.train:
        console.rule("[bold magenta]✒️ Starting Training[/bold magenta]")
        print(f'Training Settings:')
        train_start = time.time()
        print('-' * 75)
        print('\t\t🔹 CONFIGURATION (YAML)')
        print('-' * 75)
        print(yaml.dump(final_config, default_flow_style=False, indent=2))
        print('-' * 75)
        print(f'\n\n🔹 Starting Training...')
        
        print('\nModel Details')
        print('-'*50)
        model_parameters = sum(p.numel() for p in model.parameters())
        print(f"Total parameters:{model_parameters:,} \t({convert2hr(model_parameters)})") 
        print('-'*50 + '\n')
        
        trainer.train(
            max_steps=max_steps
        )
        train_end = time.time()
        training_logs = trainer.get_logs() 
        
        training_time= train_end - train_start
        
        print(f'✅ Training Done in {training_time}!')
        print(f"\t-> Training Time: {get_time_str(training_time)}\n")
        print(f'📊 Training Statistics:')
        # print(f'   Total Tokens Processed: {train_tokens:,}')
        # print(f'   Overall Tokens/Second: {overall_tps:.2f}')
    
        print('\n\n')
        print('-'*75)
        print('\t === POST TRAINING STATS ===')
        print('-'*75)
        if hasattr(training_logs, "train_loss") and training_logs["train_loss"]:
            final_loss = training_logs["train_loss"][-1]
            print(f"Final Training Loss: {final_loss:.4f}")
        if hasattr(training_logs, "val_loss") and training_logs["val_loss"]:
            final_val_loss = training_logs["val_loss"][-1]
            print(f"Final Validation Loss: {final_val_loss:.4f}")
            perplexity = torch.exp(torch.tensor(final_val_loss)).item()
            print(f"Final Validation Perplexity: {perplexity:.4f}")
            bpc = final_val_loss / torch.log(torch.tensor(2.0)).item()
            print(f"Final Validation Bits-per-Character (BPC): {bpc:.4f}")
        if hasattr(training_logs, "val_acc") and training_logs["val_acc"]:
            final_val_acc = training_logs["val_acc"][-1]
            print(f"Final Validation Accuracy: {final_val_acc:.4f}")
        if hasattr(training_logs, "tok/s") and training_logs['tok/s']:
            tps_avg = sum(training_logs['tok/s']) / len(training_logs['tok/s'])
            print(f"Average Tokens/Second: {tps_avg:.4f}")
    else:
        console.print("[bold red]❗ Not training. Use --train to start training.[/bold red]")
    
    if args.generate:
        tokenizer = get_encoder()
        print(f'🔹  Generating... ✒️ Inference:\n')

        print( '-'*100 + f'\n\tExample 1 : Starting with no context\n' + '-'*100 + \
        '\nGenerated: ' + tokenizer.decode(
            model.generate(torch.zeros((1, 1), dtype=torch.long, device=device),
                           max_pred_tokens=context_length,
                           temp=1.0,
                           top_k=50,
                           top_p=0.9,
                           )[0].tolist()) + '\n'
        )

        example_prompts = [
            "Virat Kohli is a Indian Cricket plays",
            "Once upon a time,",
            "The capital of India is New Delhi then the capital of America is",
        ]
        for it, starts_with in enumerate(example_prompts):
            contxt = torch.tensor(
                tokenizer.encode(starts_with, allowed_special = spl_tok_dict),
                dtype=torch.long
            ).unsqueeze(0).to(device)
            inference = tokenizer.decode(
                [t if t < m_cfg["vocab_size"]  else ENDOFTEXT for t in model.generate(contxt,
                               max_pred_tokens=context_length,
                               temp=1.0,
                               top_k=50,
                               top_p=0.9,
                               )[0].tolist()
                 ]
            )
            print('Infernece:\n' +'-'*100 + f'\n\tExample {it + 2} : Starting with {repr(starts_with)}\n' + '-'*100 + f'\nGenerated:{inference}\n' )


if __name__ == "__main__":
    start = datetime.datetime.now()
    main2()
    console.print('[bold green]✅ Done! Script Completed Successfuly[/bold green]')
    print(f'⌚ Script Time: {datetime.datetime.now() - start}')
    