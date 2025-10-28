import os
import logging
import tempfile
import torch
import pytest

from model.architecture import (
    MultiHeadSelfAttention,
    SwiGLU,
    SwiGLUFast,
    FeedForwardNet,
    RMSNorm,
    DecoderBlock,
    DecoderOnlyTransformer
)


log_dir = "./checkpoints/logs/tests"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "test_model.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)

@pytest.mark.parametrize("batch,seq,dim,heads", [
    (2, 16, 64, 8),
    (1, 32, 128, 8),
])
def test_mhsa_output_shape(batch, seq, dim, heads):
    x = torch.randn(batch, seq, dim)
    mhsa = MultiHeadSelfAttention(dim, heads, dropout=0.0)
    out = mhsa(x)
    assert out.shape == (batch, seq, dim)

@pytest.mark.parametrize("dim_in,dim_out", [(64, 128), (128, 256)])
def test_swiglu_variants(dim_in, dim_out):
    x = torch.randn(4, dim_in)
    for cls in (SwiGLU, SwiGLUFast):
        module = cls(dim_in, dim_out)
        out = module(x)
        assert out.shape == (4, dim_out)

@pytest.mark.parametrize("dim,hidden", [(64, 256), (128, 512)])
def test_feedforward_shapes(dim, hidden):
    x = torch.randn(3, dim)
    for non_lin in ("swiglu", "swiglufast", "gelu", "relu"):
        ffn = FeedForwardNet(dim, hidden, non_linearity=non_lin, dropout=0.0)
        out = ffn(x)
        assert out.shape == (3, dim)

def test_rmsnorm_preserves_shape_and_dtype():
    x = torch.randn(2, 10, 64).half()
    norm = RMSNorm(64)
    out = norm(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype

def test_decoder_block_forward():
    batch, seq, dim, heads, hid, nl = 2, 16, 64, 8, 256, "gelu"
    x = torch.randn(batch, seq, dim)
    block = DecoderBlock(dim, heads, hid, non_linearity=nl, dropout=0.0, use_checkpoint=False)
    out = block(x)
    assert out.shape == (batch, seq, dim)

@pytest.mark.parametrize("model_cfg", [
    {"vocab_size": 1000, "seq_len": 32, "dim": 64, "heads": 8, "layers": 2, "hid": 256},
    {"vocab_size": 5000, "seq_len": 16, "dim": 128, "heads": 8, "layers": 4, "hid": 512},
])
def test_decoder_only_transformer_forward(model_cfg):
    batch = 2
    seq_len = model_cfg["seq_len"]
    vocab_size = model_cfg["vocab_size"]
    x = torch.randint(0, vocab_size, (batch, seq_len))
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        context_length=seq_len,
        model_dimension=model_cfg["dim"],
        n_heads=model_cfg["heads"],
        Nx_blocks=model_cfg["layers"],
        ffn_hid_dim=model_cfg["hid"],
        non_linearity="gelu",
        dropout=0.0,
        tie_weights=True,
        init_std_val=0.02,
        use_checkpoint=False
    )
    logits = model(x)
    assert logits.shape == (batch, seq_len, vocab_size)

if __name__ == "__main__":
    pytest.main([__file__, "-q"])
