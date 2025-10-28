
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

def get_optimizer(model, cfg):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in ['bias', 'weight_g', 'weight_v', 'norm']):
            no_decay.append(param)
        else:
            decay.append(param)
    return AdamW(
        [
            {'params': decay, 'weight_decay': cfg['optimizer']['weight_decay']},
            {'params': no_decay, 'weight_decay': 0.0}
        ],
        lr=cfg['optimizer']['lr'],
        betas=(0.9, 0.95),
        eps=1e-8
    )

def get_scheduler(optimizer, cfg, total_steps):
    warmup = cfg['scheduler']['warmup_steps']
    def lr_lambda(step):
        if step < warmup:
            return float(step) / float(max(1, warmup))
        # return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(step - warmup) / (total_steps - warmup) * torch.pi))).item()
        cosine = 0.5 * (1.0 + torch.cos(
            torch.tensor(step - warmup, dtype=torch.float32) / (total_steps - warmup) * torch.pi
        ))
        return float(max(0.0, cosine.item()))
    return LambdaLR(optimizer, lr_lambda)
