

import argparse
from box import Box

from data.streaming_dataset import StreamingTextDataset
from model.architecture import DecoderOnlyTransformer
from model.optimizer_config import get_optimizer, get_scheduler
from training.trainer import Trainer
from utils.config import load_config
from utils.logging import setup_logging

import os
from dotenv import load_dotenv
load_dotenv() 


def main():
    parser = argparse.ArgumentParser(description="MYT-LLM Dry Run")
    parser.add_argument("--config", type=str, default=os.getenv("MYTLLM_CONFIG"), help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg, name='dry_run')

    
    ds = StreamingTextDataset(
        shard_folder=os.getenv("MYTLLM_DATA"),
        context_length=cfg.model.context_length,
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
    )

    
    cfg_dict = cfg.to_dict()
    optimizer = get_optimizer(model, cfg_dict)
    scheduler = get_scheduler(optimizer, cfg_dict, cfg.training.max_steps)

    
    trainer = Trainer(cfg, model, ds, optimizer, scheduler)
    avg_time = trainer.pretrain_check(num_iters=10)
    print(f"Average iteration time: {avg_time:.3f}s")


if __name__ == "__main__":
    main()
