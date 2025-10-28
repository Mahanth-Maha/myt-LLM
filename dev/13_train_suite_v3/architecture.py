import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dimension, n_heads, dropout=0.3):
        super().__init__()
        assert model_dimension % n_heads == 0
        self.n_heads = n_heads
        self.each_head_size = model_dimension // n_heads
        self.W_qkv = nn.Linear(model_dimension, 3 * model_dimension, bias=False)
        self.proj_attn = nn.Linear(model_dimension, model_dimension, bias=False)
        self.dropout = dropout
        # self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_qkv.weight)
        nn.init.xavier_uniform_(self.proj_attn.weight)

    def forward(self,x):
        B,T,C = x.shape

        Q,K,V = self.W_qkv(x).chunk(3, dim=-1)
        Q = Q.view(B, T, self.n_heads, self.each_head_size).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.each_head_size).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.each_head_size).transpose(1, 2)

        sdpa = F.scaled_dot_product_attention(Q,K,V,attn_mask = None, dropout_p=self.dropout, is_causal=True)
        
        output = sdpa.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj_attn(output)


class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.fc_a = nn.Linear(dim_in, dim_out, bias=False)
        self.fc_b = nn.Linear(dim_in, dim_out, bias=False)
        # nn.init.xavier_uniform_(self.fc_a.weight)
        # nn.init.xavier_uniform_(self.fc_b.weight)

    def forward(self, x):
        return F.silu(self.fc_a(x)) * self.fc_b(x)


class SwiGLUFast(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.fc = nn.Linear(dim_in, 2 * dim_out, bias=False)    
        # nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return F.silu(a) * b


class FeedForwardNet(nn.Module):
    def __init__(self, model_dimension, ffn_hid_dim, non_linearity='swiglu', dropout=0.3):
        super().__init__()
        self.non_linearity = non_linearity
        if non_linearity == 'swiglu'or non_linearity == 'swiglufast' :
            self.proj_ffn = nn.Linear(ffn_hid_dim, model_dimension)
            self.activation = SwiGLU(model_dimension, ffn_hid_dim) if non_linearity == 'swiglu' else SwiGLUFast(model_dimension, ffn_hid_dim)
        else :
            self.ln_ffn = nn.Linear(model_dimension, ffn_hid_dim, bias=False)
            self.proj_ffn = nn.Linear(ffn_hid_dim, model_dimension, bias=False)
            self.activation = nn.GELU() if non_linearity == 'gelu' else nn.ReLU()
        
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        if self.non_linearity in ('swiglu', 'swiglufast'):
            x = self.activation(x)
        else:
            x = self.activation(self.ln_ffn(x))
        return self.dropout_layer(self.proj_ffn(x))


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        orig_dtype = x.dtype
        x_float = x.to(torch.float32)
        var = x_float.pow(2).mean(-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(var + self.eps)
        return (self.weight * x_norm).to(orig_dtype)


class DecoderBlock(nn.Module):
    def __init__(self, model_dimension, n_heads, ffn_hid_dim, non_linearity='gelu', dropout=0.3, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.attn_norm = RMSNorm(model_dimension)
        self.attn = MultiHeadSelfAttention(model_dimension, n_heads, dropout)
        self.ffn_norm = RMSNorm(model_dimension)
        self.ffn = FeedForwardNet(model_dimension, ffn_hid_dim, non_linearity, dropout)

    def _forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, context_length, model_dimension, n_heads, Nx_blocks, ffn_hid_dim, non_linearity='gelu', dropout=0.3, tie_weights=True, init_std_val = 0.02, use_checkpoint=False):
        super().__init__()
        self.Nx_blocks = Nx_blocks
        self.context_length = context_length
        self.token_emb = nn.Embedding(vocab_size, model_dimension)
        self.pos_emb = nn.Embedding(context_length, model_dimension)
        self.blocks = nn.ModuleList([
            DecoderBlock(
                model_dimension, 
                n_heads, 
                ffn_hid_dim, 
                non_linearity, 
                dropout, 
                use_checkpoint
            ) for _ in range(Nx_blocks)
        ])
        self.final_norm = RMSNorm(model_dimension)
        self.lm_head = nn.Linear(model_dimension, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.token_emb.weight
        self._init_weights(init_std_val)

    def _init_weights(self, std_val = 0.02):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std_val * ((2 * self.Nx_blocks) ** -0.5))
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std_val)

    def forward(self, input_ids):
        B, T = input_ids.size()
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        for i, block in enumerate(self.blocks):
            if i >= len(self.blocks) // 2:
                x = checkpoint(block, x)
            else:
                x = block(x)
        
        x = self.final_norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, x, max_pred_tokens, temp=1.0, top_k=None, top_p=None):
        self.eval()
        target_device = x.device
        target_device = x.device
        gen = torch.Generator(device=target_device)
        
        for _ in range(max_pred_tokens):
            logits = self(x[:, -self.context_length:])
            logits = logits[:, -1, :] / temp

            if top_k is not None:
                top_k_vals, top_k_idxs = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_vals[:, [-1]]] = -float('Inf')

            if top_p is not None and 0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_mask = cumulative_probs > top_p

                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = 0

                indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                logits = logits.masked_fill(indices_to_remove, -float('Inf'))

            prob_dist = F.softmax(logits, -1)
            x = torch.cat([x, torch.multinomial(prob_dist, 1, generator=gen)], -1).to(target_device)
        self.train()
        return x
