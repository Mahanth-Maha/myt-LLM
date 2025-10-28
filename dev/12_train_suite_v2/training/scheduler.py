from torch.optim.lr_scheduler import LambdaLR
import torch

def get_scheduler(optimizer, cfg, total_steps):
    warmup_steps = cfg['scheduler']['warmup_steps']
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(
            0.0,
            0.5 * (1.0 + torch.cos(
                (step - warmup_steps) / float(total_steps - warmup_steps) * torch.pi
            ))
        )
    return LambdaLR(optimizer, lr_lambda)
