import os
import yaml
import time
import json
import math
import pytz
import argparse
import datetime
import platform

import torch
from torch.utils.data import DataLoader

import warnings
# warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from architecture import DecoderOnlyTransformer
from trainer import DecodeTransformerTrainer, TrainerConfig
from data import MultiDirectoryStreamingTextDataset, ValidationDatasetStreamer
from tokenizer import mytlmTokenizer, all_spl_toks_dict

from optimizer_config import create_optimizer, create_scheduler, get_fused_cross_entropy
from amd_gpu_setting import setup_complete_amd_environment, compile_model_amd
from utils import load_config_from_yaml, merge_configs, overrides_to_dict
from utils import convert2hr, convert2hr1, fmt_time, fmt_dt_ist, get_time_str

from rich.console import Console
from rich.panel import Panel
from rich.box import HEAVY
console = Console()

from dotenv import load_dotenv
load_dotenv() 



def parse_args():
    parser = argparse.ArgumentParser(description="Model training config overrides")
    parser.add_argument('-c',"--config_file", type=str, default=None, help="Optional additional YAML config file to override defaults")
    parser.add_argument('-z',"--model_type", type=str, default='DEBUG', help="Optional additional name of YAML config file to override defaults")
    parser.add_argument('-t',"--train", action="store_true", help="starts training")
    parser.add_argument('-g',"--generate", action="store_true", help="Toggle on to generate from model")
    parser.add_argument('-b',"--batch_size", type=int, default=None, help="Config: batch_size")
    parser.add_argument('-m',"--max_steps", type=int, default=None, help="Config: max_steps")
    parser.add_argument('-a',"--trainer.accum_steps", type=int, default=None, help="Config: accum_steps ")
    parser.add_argument('-w',"--trainer.train_time_warmup", type=int, default=None, help="Config: train_time_warmup ")
    parser.add_argument('-l',"--logging.log_dir", type=str, help="logging dir")
    parser.add_argument('-e',"--trainer.ema_decay", type=float, default=None, help="Config: ema_decay ")
    parser.add_argument('-r',"--trainer.resume_train", action="store_true", default=None, help="set it for scratch")
    parser.add_argument("--gpu.compile_approach", type=str, default=None, help="set mode")
    parser.add_argument("--data-folder-list", type=str, default=None, help="list of dirs")
    parser.add_argument("--optimizer.lr", type=float, default=None, help="Learning rate")
    return parser.parse_args()

def main():
    print(f"🧭 Torch: {torch.__version__}, CUDA: {torch.version.cuda}, Python: {platform.python_version()}")
    
    # torch.set_float32_matmul_precision('high')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    
    args = parse_args()
    
    data_folder_list = os.getenv("MYTLLM_DATA_LIST")
    default_config = os.getenv(f"MYTLLM_CONFIG_{args.model_type}")
    
    base_config = load_config_from_yaml(default_config)
    if args.data_folder_list:
        directories = json.loads(args.data_folder_list)
    else:
        directories = json.loads(data_folder_list)
        
    if args.config_file:
        user_config = load_config_from_yaml(args.config_file)
        base_config = merge_configs(base_config, user_config)
    
    cli_overrides = overrides_to_dict(args)
    final_config = merge_configs(base_config, cli_overrides)
    

    m_cfg = final_config["model"]
    batch_size = final_config["batch_size"]
    grad_accum = final_config["trainer"]['accum_steps']
    context_length = m_cfg["context_length"]
    
        
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if final_config["device"] =='cuda' else torch.device("cpu") 
    
    model_kwargs = dict(
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
    model = DecoderOnlyTransformer(**model_kwargs)
    model_params = sum(p.numel() for p in model.parameters())

    console.print(Panel.fit(
        f"[bold cyan]📌 Model:[/bold cyan] {m_cfg['name']}\n"
        f"[bold]Parameters:[/bold] {model_params:,} ({convert2hr(model_params)})\n"
        f"[bold]Context Length:[/bold] {context_length} | [bold]Batch Size:[/bold] {batch_size}\n"
        f"[bold]Device:[/bold] {device}",
        title="🧠 Model Summary",
        border_style="cyan"
    ))
    
    tokenizer = mytlmTokenizer(all_spl_toks_dict, vocab_size = m_cfg["vocab_size"])
    samples_cfg = final_config["decoding"]

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
    
    print('✅ Compiling Settings Done (Compilation may happens in Warm-up-Check)')
    
    
    max_steps = final_config["max_steps"]
    optimizer = create_optimizer(model, lr=final_config["optimizer"]["lr"])
    scheduler = create_scheduler(optimizer, max_steps, warmup_steps=final_config["scheduler"]["warmup_steps"])
    loss_fn = get_fused_cross_entropy()

    trainer_config = TrainerConfig(**final_config["trainer"])    
    trainer = DecodeTransformerTrainer(
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
        model_kwargs = model_kwargs,
        tokenizer=tokenizer,
        samples_cfg=samples_cfg,
        verbose=True,
    )

    print(f'🏃 Warming Up a little... Tuning and 🔄️ compiling your model!')
    print(f'🫸  Wait for report on Time estimates...')
    avg_time_per_opt_step = trainer.warmup_time_check_for_train(warmup_steps=final_config['trainer']['train_time_warmup'])
    
    micro_steps_per_epoch = len(train_loader)
    opt_steps_per_epoch   = math.ceil(micro_steps_per_epoch / grad_accum)

    tokens_per_batch      = batch_size * context_length
    tokens_per_opt_step   = tokens_per_batch * grad_accum
    tokens_per_epoch      = tokens_per_batch * micro_steps_per_epoch

    total_steps           = final_config['max_steps']
    total_train_tokens    = tokens_per_opt_step * total_steps

    n_params              = model_params if 'model_params' in locals() else sum(p.numel() for p in model.parameters())
    chinchilla_target     = 20 * n_params
    mahanthyalla_target   = 39 * n_params

    def shortfall_or_surplus(target_tokens, have_tokens):
        delta = have_tokens - target_tokens
        if delta >= 0:
            return f"[green]surplus +{convert2hr1(delta)} tks[/green]"
        else:
            return f"[red]shortfall -{convert2hr1(-delta)} tks[/red]"

    def recommend_settings(target_tokens):
        req_steps = math.ceil(target_tokens / tokens_per_opt_step)
        req_accum_if_fix_steps = math.ceil(max(1, target_tokens / (tokens_per_batch * total_steps)))
        return req_steps, req_accum_if_fix_steps

    chin_req_steps, chin_req_accum = recommend_settings(chinchilla_target)
    maha_req_steps,  maha_req_accum  = recommend_settings(mahanthyalla_target)

    time_per_micro_step   = avg_time_per_opt_step / max(1, grad_accum)
    time_per_opt_step     = avg_time_per_opt_step
    time_for_1M_tokens    = 1_000_000 * (time_per_opt_step / tokens_per_opt_step)
    time_one_epoch        = opt_steps_per_epoch * time_per_opt_step
    time_total_run        = total_steps * time_per_opt_step
    time_chinchilla       = chin_req_steps * time_per_opt_step
    time_mahanthyalla     = maha_req_steps  * time_per_opt_step

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
                                                                                                🍪
    • Tokens in full dataset (per epoch)      : [bold]{convert2hr(tokens_per_epoch)}[/bold] tokens ([dim]{tokens_per_epoch:,}[/dim])
    • Dataloader iterations per epoch         : [bold]{micro_steps_per_epoch:,}[/bold] iters
    • With your current accumulation:
        - Batch Size                          : [bold]{batch_size:,}[/bold]
        - Iteration steps per epoch           : [bold]{micro_steps_per_epoch:,}[/bold]
        
        - Accumulation Steps                  : [bold]{final_config['trainer']['accum_steps']:,}[/bold]
        - Optimizer steps per epoch           : [bold]{opt_steps_per_epoch:,}[/bold]
    """
    # """
    # [dim]Legend:[/dim]
    #     • [bold]Iteration[/bold] = one dataloader [micro] batch → “micro-step”
    #     • [bold]Optimizer step[/bold] = after accumulating [bold]{grad_accum}[/bold] iterations, we step the optimizer
    # """

    box2 = f"""
    [bold magenta]TOKENS & BATCHING[/bold magenta]
    
    • Given batch size                                     : [bold]{batch_size}[/bold]
    • Gradient accumulation steps                          : [bold]{grad_accum}[/bold]
    • Effective batch size (BATCH_SIZE)                    : [bold]{grad_accum * batch_size}[/bold]

    • Tokens per iteration (one dataloader batch) (micro)  : [bold]{tokens_per_batch:,}[/bold]
    • Tokens per optimizer step (after accum;BATCH_SIZE)   : [bold]{convert2hr(tokens_per_opt_step)}[/bold]
                                                                        ([dim]{tokens_per_opt_step:,}[/dim])
    • Planned steps in this run                            : x [bold]{total_steps:,}[/bold]
                                                           =================    
    • [bold]Total train tokens in this run[/bold]                       : = [bold]{convert2hr(total_train_tokens)}[/bold]
                                                                        ([dim]{total_train_tokens:,}[/dim])

    [bold magenta]Targets[/bold magenta]
    • Model size                        : [bold]{convert2hr(n_params)}[/bold]  ([dim]{n_params:,}[/dim])
    • Training tokens                   : [bold]{convert2hr(total_train_tokens)}[/bold]  ([dim]{total_train_tokens:,}[/dim])
    • Chinchilla target  (20× params)   : [bold]{convert2hr(chinchilla_target)}[/bold] (needed)\n\t\t\t\t\t → {shortfall_or_surplus(chinchilla_target, total_train_tokens)}
    • Mahanthyalla target (39× params)  : [bold]{convert2hr(mahanthyalla_target)}[/bold] (needed)\n\t\t\t\t\t → {shortfall_or_surplus(mahanthyalla_target, total_train_tokens)}

    [bold magenta]To hit targets (keeping batch={batch_size}, context={context_length})[/bold magenta]
    • Chinchilla:
        - Required optimizer steps                      : [bold]{chin_req_steps:,}[/bold]
        - Or keep steps = {total_steps:7,} → set accum           : [bold]{chin_req_accum}[/bold]
    • Mahanth Yalla:
        - Required optimizer steps                      : [bold]{maha_req_steps:,}[/bold]
        - Or keep steps = {total_steps:7,} → set accum           : [bold]{maha_req_accum}[/bold]
    """

    box3 = f"""
    [bold green]TIME & ETAs[/bold green]
                                                                                ⌛
    [bold]Per-unit times[/bold]
    • Time per iteration (one dataloader [micro] batch)\t: [bold]{time_per_micro_step:.4f} s[/bold]
    • Time per optimizer step (after {grad_accum:2} iters)         \t: [bold]{time_per_opt_step:.4f} s[/bold]
    • Time for 1M tokens (at current mix)              \t: [bold]{fmt_time(time_for_1M_tokens)}[/bold]

    [bold]Larger spans[/bold]
    • One full epoch time       \t\t: [bold]{fmt_time(time_one_epoch)}[/bold]
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
        console.rule("[bold magenta]✒️  Starting Training[/bold magenta]")
        print(f'Training info:')
        print(f"""
              Estimated Timings
                        Estimated Time of RUN  🏃 💨 : {fmt_time(time_total_run)}
                        Starting at ✍️  : {fmt_dt_ist(current_fixed)}
                        May End  at 🏁 : {fmt_dt_ist(eta_run)}
              """)
        print('\nModel Details')
        print('-'*50)
        model_parameters = sum(p.numel() for p in model.parameters())
        print(f"Total parameters:{model_parameters:,} \t({convert2hr(model_parameters)})") 
        print('-'*50 + '\n')
        
        train_start = time.time()
        print('-' * 75)
        print(f'🔹 CONFIGURATION (YAML)')
        print('-' * 75)
        print(yaml.dump(final_config, default_flow_style=False, indent=2))
        print('-' * 75)
        print(f'\n\n🔹 Starting Training...')
        
        trainer.train(
            max_steps=max_steps
        )
        train_end = time.time()
        
        training_time= train_end - train_start
        
        print(f'✅ Training Done in {training_time}!')
        print(f"\t-> Training Time: {get_time_str(training_time)}\n")
        
        # print(f'📊 Training Statistics:')
        # # print(f'   Total Tokens Processed: {train_tokens:,}')
        # # print(f'   Overall Tokens/Second: {overall_tps:.2f}')
    
        # print('\n\n')
        # print('-'*75)
        # print('\t === POST TRAINING STATS ===')
        # print('-'*75)
        # training_logs = trainer.get_logs() 
        # if hasattr(training_logs, "train_loss") and training_logs["train_loss"]:
        #     final_loss = training_logs["train_loss"][-1]
        #     print(f"Final Training Loss: {final_loss:.4f}")
        # if hasattr(training_logs, "val_loss") and training_logs["val_loss"]:
        #     final_val_loss = training_logs["val_loss"][-1]
        #     print(f"Final Validation Loss: {final_val_loss:.4f}")
        #     perplexity = torch.exp(torch.tensor(final_val_loss)).item()
        #     print(f"Final Validation Perplexity: {perplexity:.4f}")
        #     bpc = final_val_loss / torch.log(torch.tensor(2.0)).item()
        #     print(f"Final Validation Bits-per-Character (BPC): {bpc:.4f}")
        # if hasattr(training_logs, "val_acc") and training_logs["val_acc"]:
        #     final_val_acc = training_logs["val_acc"][-1]
        #     print(f"Final Validation Accuracy: {final_val_acc:.4f}")
        # if hasattr(training_logs, "tok/s") and training_logs['tok/s']:
        #     tps_avg = sum(training_logs['tok/s']) / len(training_logs['tok/s'])
        #     print(f"Average Tokens/Second: {tps_avg:.4f}")
        
    else:
        console.print("[bold red]❗ Not training. Use --train to start training.[/bold red]")
    
    if args.generate:
        print(f'🔹  Generating... ✒️  Inference:\n')
        print( '-'*100 + f'\n\tExample 1 : Starting with no context\n' + '-'*100 + \
        '\nGenerated: ' + tokenizer.generate( 
                            model,
                            prompt='',
                            max_pred_tokens=context_length,
                            temp=1.0,
                            top_k=50,
                            top_p=0.9,
                            device=device,
                           ) + '\n'
        )
        example_prompts = [
            "The Theory of Relativity was proposed by",
            "How to make a simple paper airplane:\n1.",
            "Why do eclipses not happen every month?\nBecause",
            "The capital of India is New Delhi then the capital of France is",
            "The man couldn't lift his son because he was so weak. Who was weak?",
            "Translate English to French:\nEnglish: I love learning new languages.\nFrench:",
            "This is an Explain of how photosynthesis works in simple terms: Photosynthesis",
            "Write a Python function that returns the factorial of a number.\n```python\ndef"
            "Virat Kohli is widely considered by many as one of the greatest batsmen in the history of cricket",
            "Once upon a time, in a small village near the mountains, a young inventor discovered a strange machine buried under the old oak tree.",
            "A train leaves the station at 3:00 PM and travels at 60 km/h. Another train leaves the same station at 4:00 PM and travels at 90 km/h in the same direction. At what time will the second train catch up?",
            "Summarize the following paragraph in one sentence:\n\nArtificial intelligence refers to systems or machines that mimic human intelligence to perform tasks and can iteratively improve themselves based on the information they collect. Examples of AI include chatbots, recommendation systems, and autonomous vehicles.\n Summary:",
        ]
        for it, starts_with in enumerate(example_prompts):
            inference = tokenizer.generate( 
                            model,
                            prompt=starts_with,
                            max_pred_tokens=context_length,
                            temp=1.0,
                            top_k=50,
                            top_p=0.9,
                            device=device,
                        )
            print('Infernece:\n' +'-'*100 + f'\n\tExample {it + 2} : Starting with {repr(starts_with)}\n' + '-'*100 + f'\nGenerated:{inference}\n' )

        context_length_test_prompts = [
            "The capital of France is",
            "The capital of France is Paris, which is known for its famous landmarks.",
            "The capital of France is Paris, which is known for its famous landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, attracting millions of tourists every year.",
            "The capital of France is Paris, which is known for its famous landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, attracting millions of tourists every year, and serving as a global center for art, fashion, culture, and history, making it one of the most visited cities in the world.",
            "The capital of France is Paris, which is known for its famous landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, attracting millions of tourists every year, and serving as a global center for art, fashion, culture, and history, making it one of the most visited cities in the world, while also playing a major role in European politics, being the site of numerous historical events, treaties, and cultural movements that have shaped not only France but also the modern world as we know it."
        ]
        print('\n\n')
        print('-'*100)
        print(f'Context length Stress Test:')
        print('-'*100)
        for it, starts_with in enumerate(context_length_test_prompts):
            inference = tokenizer.generate( 
                            model,
                            prompt=starts_with,
                            max_pred_tokens=context_length,
                            temp=1.0,
                            top_k=50,
                            top_p=0.9,
                            device=device,
                        )
            print('Infernece:\n' +'-'*100 + f'\n\tContext Test {it + 1} : Starting with {repr(starts_with)}\n' + '-'*100 + f'\nGenerated:{inference}\n' )
        print('\n')


if __name__ == "__main__":
    start = datetime.datetime.now()
    main()
    console.print('[bold green]✅ Done! Script Completed Successfuly[/bold green]')
    print(f'⌚ Script Time: {datetime.datetime.now() - start}')
    