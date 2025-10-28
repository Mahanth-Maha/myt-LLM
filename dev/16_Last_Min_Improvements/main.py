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

from tokenizer import get_encoder, convert2hr, spl_tok_dict
from optimizer_config import get_metrics_from_loader, create_optimizer, create_scheduler, get_fused_cross_entropy
from amd_gpu_setting import setup_complete_amd_environment, compile_model_amd
from trainer import GeneralTrainer 
# from architecture import DecoderOnlyTransformer
from architecture2 import DecoderOnlyTransformer
from data2 import StreamingTextDataset,MultiDirectoryStreamingTextDataset, ValidationDatasetStreamer,RobustValidationDataset


from dotenv import load_dotenv

load_dotenv() 
data_folder = os.getenv("MYTLLM_DATA")
data_folder_list = os.getenv("MYTLLM_DATA_LIST")
default_config = os.getenv("MYTLLM_CONFIG")



def load_config_from_yaml(path):
    """Load configuration dictionary from a YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def merge_configs(base, override):
    """Recursively merge override config into base config, creating missing keys."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = merge_configs(base[k], v)
        else:
            base[k] = v
    return base


def parse_args():
    """Parse command-line arguments for configuration overrides."""
    parser = argparse.ArgumentParser(description="Model training config overrides")
    parser.add_argument("--config_file", type=str, default=None, help="Optional additional YAML config file to override defaults")
    parser.add_argument('-t',"--train", action="store_true", help="starts training")
    parser.add_argument('-s',"--training.resume_train", action="store_false", help="set it for scratch")
    parser.add_argument('-g',"--generate", action="store_true", help="Toggle on to generate from model")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--over-fit-check", action="store_true", help="Overfit Checking")
    parser.add_argument("--optimizer.lr", type=float, help="Learning rate")
    parser.add_argument("--model.num_layers", type=int, help="Number of model layers")
    parser.add_argument("--logging.log_dir", type=str, help="logging dir")
    parser.add_argument("--scheduler.type", type=str, help="Scheduler type")
    return parser.parse_args()


def overrides_to_dict(args):
    """Convert CLI overrides from args Namespace into nested dictionary."""
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


def main():
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


    m_cfg = final_config['model']
    batch_size=final_config["batch_size"]
    context_length = m_cfg['context_length']
    
    # train_dataset = StreamingTextDataset(data_folder, context_length=context_length)
    # val_dataset_streamer = ValidationDatasetStreamer(data_folder, context_length=context_length)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    # val_loader = val_dataset_streamer.get_val_loader(batch_size=batch_size)
    
    directory_list = json.loads(data_folder_list)
    train_dataset = MultiDirectoryStreamingTextDataset(
        directory_list, 
        context_length=context_length,
        )
    val_dataset_streamer = ValidationDatasetStreamer(
        directory_list, 
        context_length=context_length
        )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = val_dataset_streamer.get_val_loader(
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        shuffle=False
    )
    

    print(f"Train samples: {len(train_dataset) * context_length} ({convert2hr(len(train_dataset) * context_length)}), Val samples: {len(val_loader.dataset)} ({convert2hr(len(val_loader.dataset))})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = m_cfg["vocab_size"]
    print(f'Device Set to:{device}')
    print(f'Vocab size :{vocab_size}\n')
    print(f'⚠️ Model Architectural Changes: (architecture v2)')
    print(f'\tActivation: Default Non linearity is set to `SwiGLU` fused with FFN linear layer')
    print(f'\tAttention : now supports GQA keeping `n_kv_heads` to -1 sets back to single KV MHA')
    print(f'\t')
    
    model = DecoderOnlyTransformer(
        vocab_size=m_cfg["vocab_size"],
        context_length=m_cfg["context_length"],
        model_dimension=m_cfg["dimension"],
        n_heads=m_cfg["n_heads"],
        Nx_blocks=m_cfg["num_layers"],
        ffn_hid_dim=m_cfg["hidden_dimension"],
        # non_linearity=m_cfg["non_linearity"],
        n_kv_heads =m_cfg["n_kv_heads"] if m_cfg["n_kv_heads"] != -1 else None,
        dropout=m_cfg["dropout"],
        tie_weights=m_cfg.get("tie_weights", True),
        use_checkpoint=final_config["training"].get("use_checkpoint", False),
        checkpoint_ratio=final_config["training"].get("checkpoint_ratio", 0.5),
    )
    
    print('\nModel Created:')
    print('-'*50)
    model_parameters = sum(p.numel() for p in model.parameters())
    print(f"Total parameters:{model_parameters:,} \t({convert2hr(model_parameters)})") 
    print('-'*50 + '\n')
    chinchila_rule = 20
    print(f"Rule : To train we need {chinchila_rule} tkns/parm -> Required tokens = {model_parameters * chinchila_rule:,} \t({convert2hr(model_parameters * chinchila_rule)})\n\n") 
    

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
    
    
    num_epochs=final_config['training']['max_epochs']
    max_steps=final_config['training']['max_steps']
    force_epochs=final_config['training']['force_epochs']

    optimizer = create_optimizer(model, lr=1.5e-4)
    scheduler = create_scheduler(optimizer, max_steps, warmup_steps=2500)
    
    loss_fn = get_fused_cross_entropy()

    trainer = GeneralTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        log_dir=final_config['logging']['log_dir'],
        scheduler=scheduler,
        live_plot=False, # set True for inline plots in notebooks
        milestone=final_config['milestones']['interval_steps'],
        keep_last=final_config['checkpoint']['backups']['keep_last'],
        metric_function = None,
        precision_config=training_precision
    )

    avg_time_per_iter = trainer.warmup_time_check_for_train()
    print('-'*75)
    print('\t\tTRAINING STATS')
    print('-'*75)
    step_per_epoch = data_len = len(train_loader)
    
    time_per_epoch = avg_time_per_iter * data_len
    total_epoch_time = time_per_epoch * num_epochs

    tokens_per_batch = context_length * batch_size
    tokens_per_epoch = tokens_per_batch * data_len
    
    print(f'Max epochs set to\t\t: {num_epochs:14.0f} epochs')
    print(f'Steps per epoch\t\t\t: {step_per_epoch:14.0f} steps')
    print(f"batch_size set to\t\t: {batch_size:14.0f} batch size")
    # print(f'No of steps in epoch\t\t: {step_per_epoch:14.0f} steps')
    print(f'Avg time per step\t\t: {avg_time_per_iter:14.4f} seconds')
    print(f"Estimated time per epoch\t: {time_per_epoch:14.4f} seconds")
    
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
    
    print(f"Estimated time for {num_epochs} epochs\t: {get_time_str(total_epoch_time)}")
    

    # Time to process 1M tokens
    tokens_1M = 1_000_000
    iters_for_1M = tokens_1M / tokens_per_batch
    time_for_one_token = avg_time_per_iter / tokens_per_batch
    time_for_1M_tokens = avg_time_per_iter * iters_for_1M
    print('-' * 75)
    print(f"batch_size: {batch_size} \t|\tTokens per batch: {tokens_per_batch} ({convert2hr(tokens_per_batch)})")
    if force_epochs:
        total_time = total_epoch_time
        print(f"⚠️  force_epochs is True: Using total_epoch_time ({total_epoch_time:.2f} seconds)")
        train_tokens = tokens_per_epoch * num_epochs
        print(f"Tokens for {num_epochs} epochs (to be train tokens): {train_tokens} ({convert2hr(train_tokens)})")
        print(f"Time to process 1M tokens\t\t\t: {time_for_1M_tokens:14.2f} seconds ({get_time_str(time_for_1M_tokens)})")
        time_for_t_tokens = train_tokens * time_for_one_token
        print(f"Estimated total time to process train tokens\t: {total_time:14.2f} seconds ({get_time_str(total_time)})")
        # print(f"Estimated total time for {num_epochs} epochs: {time_for_t_tokens:.2f} seconds ({get_time_str(time_for_t_tokens)})")
    else:
        total_time = max_steps * avg_time_per_iter
        print(f"⚠️  force_epochs is False: Using max_steps ({max_steps} * {avg_time_per_iter:.2f} = {total_time:.2f} seconds)")
        train_tokens = tokens_per_batch * max_steps
        print(f"Tokens for max_steps (to be train tokens): {train_tokens} ({convert2hr(train_tokens)})")
        print(f"Time to process 1M tokens\t\t\t: {time_for_1M_tokens:14.2f} seconds ({get_time_str(time_for_1M_tokens)})")
        time_for_t_tokens = train_tokens * time_for_one_token
        print(f"Estimated total time to process train tokens\t: {total_time:14.2f} seconds ({get_time_str(total_time)})")
        # print(f"Estimated total time for {max_steps} steps: {time_for_t_tokens:.2f} seconds ({get_time_str(time_for_t_tokens)})")

    
    now = datetime.datetime.now()
    estimated_completion = now + datetime.timedelta(seconds=total_time)
    print('\n\n')
    print('-'*75)
    print(f"\t\tSCRIPT COMPLETION TIME - STATS")
    print('-'*75)
    print(f"Estimated Total time (for {max_steps} Steps): {total_time:.2f} seconds (or) {math.ceil(total_time/3600)} Hours")
    print(f"Estimated Total time (for {max_steps} Steps): {get_time_str(total_time)}")
    print(f"Date and Time (in PC) Now \t\t\t: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    ist = pytz.timezone('Asia/Kolkata')
    now_ist = now.astimezone(ist)
    eta_ist = estimated_completion.astimezone(ist)
    print(f"Current Date/Time (in IST)\t\t\t: {now_ist.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"ETA (Results) Date/Time (in IST)\t: {eta_ist.strftime('%Y-%m-%d %H:%M:%S')}")
    print('-'*75)
    print('\n\n')
    
    if args.over_fit_check: 
        data_X = np.random.randint(0, args.vocab_size, (batch_size, context_length))
        data_y = np.random.randint(0, args.vocab_size, (batch_size * context_length))
        random_X = torch.tensor(data_X, dtype=torch.long, device=device)
        random_y = torch.tensor(data_y, dtype=torch.long, device=device)
        # TODO 
        pass
    
    if args.train:
        print(f'Training Settings:')
        train_start = time.time()
        print('-' * 75)
        print('\t\tCONFIGURATION (YAML)')
        print('-' * 75)
        print(yaml.dump(final_config, default_flow_style=False, indent=2))
        print('-' * 75)
        print(f'\n\nStarting Training...')
        print(f'\tTokens Processing \t: {convert2hr(train_tokens)}')
        print(f'\tEstimatied time \t:{get_time_str(total_time)}')
        
        print('Model Created\n')
        print('-'*50)
        model_parameters = sum(p.numel() for p in model.parameters())
        print(f"Total parameters:{model_parameters:,} \t({convert2hr(model_parameters)})") 
        print('-'*50 + '\n')
        chinchila_rule = 20
        print(f"Rule : To train we need {chinchila_rule} tkns/parm -> Required tokens in dataset = {model_parameters * chinchila_rule:,} \t({convert2hr(model_parameters * chinchila_rule)})") 
        print(f"Dataset has {data_len} steps ({convert2hr(data_len)})")
        print(f"Total samples in dataset: {len(train_dataset)} ({convert2hr(len(train_dataset))})")
        print(f"Total tokens in dataset: {len(train_dataset) * context_length} ({convert2hr(len(train_dataset) * context_length)})")
        print(f"Total tokens will be Trainned: {train_tokens} ({convert2hr(train_tokens)})")
        delta = model_parameters * chinchila_rule - train_tokens
        if delta > 0:
            print(f"❓ Needed more tokens ! ( short off : {delta }\t[{convert2hr(delta)}] )") 
        if delta < 0:
            print(f"🤞 Good enough data ! ( surplus data : {-delta }\t[{convert2hr(-delta)}] )") 
            
        print(f'🔹 Training ...')
        if final_config['training']['resume_train']:
            print(f'⚠️ Training from scratch ...')
        st = datetime.datetime.now()
        trainer.train(
            num_epochs=final_config['training']['max_epochs'],
            val_iters=final_config['training']['eval_steps'],
            save_steps=final_config['training']['save_steps'],
            max_steps=final_config['training']['max_steps'],
            force_epochs=final_config['training']['force_epochs'],
            resume_train=final_config['training']['resume_train'],
        )
        train_end = time.time()
        et = datetime.datetime.now()
        train_time = et - st
        training_logs = trainer.get_logs() 
        total_duration = train_time.total_seconds()
        overall_tps = train_tokens / total_duration if total_duration > 0 else 0
        
        training_time= train_end - train_start
        
        print(f'✅ Training Done in {train_time}!')
        print(f'📊 Training Statistics:')
        print(f'   Total Tokens Processed: {train_tokens:,}')
        print(f'   Overall Tokens/Second: {overall_tps:.2f}')
        
        print(f'\nActual Training Time: {training_time} seconds')
        print(f"Actual Training Time: {get_time_str(training_time)}\n")
        delta = training_time - total_time
        if delta < 0 :
            print(f'\t💁 Finished faster than expected!\n')
        else:
            print(f'\t🤷 Finished slower than expected!\n')
        print(f"Difference: {delta:.2f} secs")
        print(f"Difference Time: {get_time_str(delta)}")
        
        
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
        print(f'❗ [Not Training] Set `--train` option to start the actual training !\n')
    
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
            "Padma Vibhush",
            "Virat Koh",
            # "Virat Koh",
            # "Virat Koh",
            # "Virat Koh",
            # "Virat Koh",
            "As with previous Valkyira Chronic",
        ]
        for it, starts_with in enumerate(example_prompts):
            contxt = torch.tensor(
                tokenizer.encode(starts_with, allowed_special = spl_tok_dict),
                dtype=torch.long
            ).unsqueeze(0).to(device)
            inference = tokenizer.decode(
                model.generate(contxt,
                               max_pred_tokens=context_length,
                               temp=1.0,
                               top_k=50,
                               top_p=0.9,
                               )[0].tolist()
            )
            print('Infernece:\n' +'-'*100 + f'\n\tExample {it + 2} : Starting with {repr(starts_with)}\n' + '-'*100 + f'\nGenerated:{inference}\n' )

    

if __name__ == "__main__":
    start = datetime.datetime.now()
    
    main()
    
    print('✅ Done! Script Completed Successfuly')
    print(f'⌚ Script Time: {datetime.datetime.now() - start}')
    