import torch
import torch.nn.functional as F


def top_k_top_p_sampling(logits,top_k=None, top_p=None, temperature=1.0):
    logits = logits / temperature

    if top_k is not None:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        cutoff = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < cutoff, torch.full_like(
            logits, -float('Inf')), logits)

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        mask = cumulative > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, -float('Inf'))
        logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1)
    return next_tokens
