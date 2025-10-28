# scripts/generate_samples.py

import argparse
import torch
import yaml
from box import Box
from model.architecture import DecoderOnlyTransformer
from utils.config import load_config
from utils.logging import setup_logging
from model.tokenizer import get_encoder
import os
from dotenv import load_dotenv
load_dotenv() 


def main():
    parser = argparse.ArgumentParser(description="Generate Samples with MYT-LLM")
    parser.add_argument("--config", type=str, default=os.getenv("MYTLLM_CONFIG"), help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model.pt")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Initial text prompt")
    parser.add_argument("--max_tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--temp", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg, name='samples')

    m_cfg = cfg.model

    model = DecoderOnlyTransformer(
        vocab_size=m_cfg.vocab_size,
        context_length=m_cfg.context_length,
        model_dimension=m_cfg.dimension,
        n_heads=m_cfg.n_heads,
        Nx_blocks=m_cfg.num_layers,
        ffn_hid_dim=m_cfg.hidden_dimension,
        non_linearity=m_cfg.non_linearity,
        dropout=0.0
    )
    
    encoder = get_encoder()
    
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    logger = setup_logging(cfg.to_dict(), name="samples")

    if args.prompt:
        user_prompt_ids = encoder.encode(args.prompt)
        prompt_ids = torch.tensor(user_prompt_ids, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        output_tkns = model.generate(
            prompt_ids,
            max_pred_tokens=args.max_tokens,
            temp=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    output = ''
    for i, sample in enumerate(output_tkns):
        output = encoder.decode(sample.tolist())
        print(f"=== Sample {i+1} ===")
        print(output)
        print()
    
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Generated IDs: {output}")

if __name__ == "__main__":
    main()
