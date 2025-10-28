import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, base = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # T_i = base^(-2i/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._precompute_cos_sin(max_seq_len)
    
    def _precompute_cos_sin(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    def rotate_half(self, x) :
        # Rotate half dimensions  [x1, x2] -> [-x2, x1]
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q, k, start_pos = 0):
        seq_len = q.shape[1]
        
        if start_pos + seq_len > self.max_seq_len:
            self._precompute_cos_sin(start_pos + seq_len)
            self.max_seq_len = start_pos + seq_len
        
        cos = self.cos_cached[start_pos:start_pos + seq_len]
        sin = self.sin_cached[start_pos:start_pos + seq_len]
        
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        
        # Apply rotation: x * cos + rotate_half(x) * sin
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_embed, k_embed

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dimension, context_length, n_heads, n_kv_heads=None, dropout=0.3):
        super().__init__()
        assert model_dimension % n_heads == 0, "model_dimension must be divisible by n_heads"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.each_head_size = model_dimension // n_heads
        assert self.each_head_size % 2 == 0, "each_head_size must be even for RoPE"

        self.dropout = dropout

        total_proj_dim = (n_heads + 2 * self.n_kv_heads) * self.each_head_size
        self.W_qkv = nn.Linear(model_dimension, total_proj_dim, bias=False)


        self.proj_attn = nn.Linear(model_dimension, model_dimension, bias=False)

        self.rope = RotaryPositionalEmbedding(
            dim=self.each_head_size,
            max_seq_len=context_length,
        )

        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        self.replication_factor = n_heads // self.n_kv_heads
        self.q_end = self.n_heads * self.each_head_size
        self.k_end = self.q_end + self.n_kv_heads * self.each_head_size

    def forward(self, x, start_pos=0):
        B, T, C = x.shape

        QKV = self.W_qkv(x)


        Q = QKV[:, :, :self.q_end]
        K = QKV[:, :, self.q_end: self.k_end]
        V = QKV[:, :, self.k_end:]

        Q = Q.view(B, T, self.n_heads, self.each_head_size)
        K = K.view(B, T, self.n_kv_heads, self.each_head_size)
        V = V.view(B, T, self.n_kv_heads, self.each_head_size)

        Q_rope, K_rope = self.rope(Q, K, start_pos)
        
        Q = Q_rope.transpose(1, 2)
        K = K_rope.transpose(1, 2)
        V = V.transpose(1, 2)     

        if self.n_kv_heads < self.n_heads:
            K = K.repeat_interleave(self.replication_factor, dim=1)
            V = V.repeat_interleave(self.replication_factor, dim=1)

        sdpa = F.scaled_dot_product_attention(Q, K, V, 
                                              attn_mask = None, 
                                              dropout_p=self.dropout if self.training else 0.0,
                                              is_causal=True
                                              )
        
        output = sdpa.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj_attn(output)

class FusedFNNSwiGLU(nn.Module):
    def __init__(self, model_dimension, ffn_hid_dim, dropout=0.3):
        super().__init__()
        self.proj_silu = nn.Linear(model_dimension, 2 * ffn_hid_dim, bias=False)    
        self.proj_ffn = nn.Linear(ffn_hid_dim, model_dimension, bias=False)    
        self.dropout_layer = nn.Dropout(dropout)
        # nn.init.xavier_uniform_(self.proj_silu.weight)
        # nn.init.xavier_uniform_(self.proj_ffn.weight)

    def forward(self, x):
        a, b = self.proj_silu(x).chunk(2, dim=-1)
        x = F.silu(a) * b
        return self.dropout_layer(self.proj_ffn(x))


# class FeedForwardNet(nn.Module):
#     def __init__(self, model_dimension, ffn_hid_dim, non_linearity='swiglu', dropout=0.3):
#         super().__init__()
#         self.non_linearity = non_linearity
#         if non_linearity == 'swiglu'or non_linearity == 'swiglufast' :
#             self.proj_ffn = nn.Linear(ffn_hid_dim, model_dimension)
#             self.activation = SwiGLU(model_dimension, ffn_hid_dim) if non_linearity == 'swiglu' else SwiGLUFast(model_dimension, ffn_hid_dim)
#         else :
#             self.ln_ffn = nn.Linear(model_dimension, ffn_hid_dim, bias=False)
#             self.proj_ffn = nn.Linear(ffn_hid_dim, model_dimension, bias=False)
#             self.activation = nn.GELU() if non_linearity == 'gelu' else nn.ReLU()
        

#     def forward(self, x):
#         if self.non_linearity in ('swiglu', 'swiglufast'):
#             x = self.activation(x)
#         else:
#             x = self.activation(self.ln_ffn(x))
#         return self.dropout_layer(self.proj_ffn(x))


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
    def __init__(self, model_dimension, context_length, n_heads, ffn_hid_dim, n_kv_heads =None, dropout=0.3):
        super().__init__()
        self.attn_norm = RMSNorm(model_dimension)
        self.attn = MultiHeadSelfAttention(model_dimension, context_length, n_heads, n_kv_heads, dropout)
        self.ffn_norm = RMSNorm(model_dimension)
        # self.ffn = FeedForwardNet(model_dimension, ffn_hid_dim, non_linearity, dropout)
        self.ffn = FusedFNNSwiGLU(model_dimension, ffn_hid_dim, dropout)
        

    def forward(self, x, start_pos = 0):
        x = x + self.attn(self.attn_norm(x),start_pos)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, context_length, model_dimension, n_heads, Nx_blocks, ffn_hid_dim, n_kv_heads =None, dropout=0.3, tie_weights=True, init_std_val = 0.02, use_checkpoint=False, checkpoint_ratio = 0.5):
        super().__init__()
        self.Nx_blocks = Nx_blocks
        self.use_checkpoint = use_checkpoint
        self.ckpt_start = int(Nx_blocks * checkpoint_ratio)
        self.context_length = context_length
        self.token_emb = nn.Embedding(vocab_size, model_dimension)
        # self.pos_emb = nn.Embedding(context_length, model_dimension)
        self.blocks = nn.ModuleList([
            DecoderBlock(
                model_dimension, 
                context_length,
                n_heads, 
                ffn_hid_dim, 
                n_kv_heads, 
                dropout
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

    def forward(self, input_ids, start_pos = 0):
        B, T = input_ids.size()
        # positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        # x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.token_emb(input_ids)
        for i, block in enumerate(self.blocks):
            if self.training and self.use_checkpoint and i >= self.ckpt_start:
                x = checkpoint(lambda t: block(t, start_pos), x, use_reentrant=False)
            else:
                x = block(x, start_pos)
        
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
