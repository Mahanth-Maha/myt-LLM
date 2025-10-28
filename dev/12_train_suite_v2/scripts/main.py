import argparse
import yaml
from box import Box
import torch

from data.streaming_dataset import StreamingTextDataset
from data.streaming_dataset import get_val_loader
from model.architecture import DecoderOnlyTransformer
from model.optimizer_config import get_optimizer, get_scheduler
from training.trainer import Trainer
from utils.config import load_config
from utils.logging import setup_logging


import os
from dotenv import load_dotenv
load_dotenv() 

def main():
    parser = argparse.ArgumentParser(description="MYT-LLM Pretraining")
    parser.add_argument("--config", type=str, default=os.getenv("MYTLLM_CONFIG"), help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg, name='train')

    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    data_folder = os.getenv("MYTLLM_DATA")
    ds = StreamingTextDataset(shard_folder=data_folder, context_length=cfg.model.context_length)
    val_loader = get_val_loader(
        shard_folder=data_folder,
        context_length=cfg.model.context_length,
        batch_size=cfg.batch_size,
        num_workers=cfg.dataloader_settings.num_workers,
        pin_memory=cfg.dataloader_settings.pin_memory
    )

    m_cfg = cfg.model
    model = DecoderOnlyTransformer(
        vocab_size=m_cfg.vocab_size,
        context_length=m_cfg.context_length,
        model_dimension=m_cfg.dimension,
        n_heads=m_cfg.n_heads,
        Nx_blocks=m_cfg.num_layers,
        ffn_hid_dim=m_cfg.hidden_dimension,
        non_linearity=m_cfg.non_linearity,
        dropout=m_cfg.dropout,
        tie_weights=m_cfg.get("tie_weights", True),
        use_checkpoint=cfg.training.get("use_checkpoint", False),
    )

    cfg_dict = cfg.to_dict()
    optimizer = get_optimizer(model, cfg_dict)
    total_steps = cfg.training.max_steps
    scheduler = get_scheduler(optimizer, cfg_dict, total_steps)

    trainer = Trainer(cfg, model, ds, optimizer, scheduler, val_dataset=val_loader.dataset)
    trainer.train_loop()

if __name__ == "__main__":
    main()