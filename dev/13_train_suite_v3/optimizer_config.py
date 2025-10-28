
import torch
import math
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

@torch.no_grad()
def get_metrics_from_loader(
    model, 
    val_loader, 
    loss_fn, 
    device, 
    val_iters = 100,
    precision_config = None
):
    if precision_config is None:
        precision_config = {'autocast_enabled': False, 'dtype': torch.float32}
    
    model.eval()
    total_loss = total_correct = total_tokens = 0
    val_iter = iter(val_loader)
    for i in range(val_iters):
        try:
            X, Y = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            try:
                X, Y = next(val_iter)
            except StopIteration:
                break
        
        X, Y = X.to(device), Y.to(device)
        
        if precision_config.get('autocast_enabled', False):
            with torch.amp.autocast('cuda', dtype=precision_config.get('dtype', torch.bfloat16)):
                logits = model(X)
                loss = loss_fn(logits, Y)
        else:
            logits = model(X)
            loss = loss_fn(logits, Y)
        
        total_loss += loss.item() * Y.numel()
        total_tokens += Y.numel()
        preds = logits.argmax(dim=-1)
        total_correct += (preds == Y).sum().item()

    avg_loss = (total_loss / total_tokens) if total_tokens > 0 else 0.0
    ppl = math.exp(avg_loss) if avg_loss < 10 else float('inf')
    acc = (total_correct / total_tokens) if total_tokens > 0 else 0.0
    bpc = avg_loss / math.log(2)

    model.train()
    return {"loss": avg_loss, "perplexity": ppl, "accuracy": acc, "bpc": bpc}

def create_optimizer(model, lr=1.5e-4):
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if (len(param.shape) == 1 or 
                name.endswith('.bias') or 
                'norm' in name.lower() or 
                'embed' in name.lower()):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,         
        weight_decay=0.1  
    )
    
    print(f"📜 AdamW optimizer: LR={lr}, β1=0.9, β2=0.95, WD=0.1, ε=1e-8")
    print(f"\tParameters with decay: {len(decay_params):,}")
    print(f"\tParameters without decay: {len(no_decay_params):,}")
    
    return optimizer


def create_scheduler(optimizer, total_steps, warmup_steps=2500):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return 0.01 + 0.99 * cosine_decay
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"📜 Cosine scheduler: {total_steps:,} total steps, {warmup_steps:,} warmup")
    print(f"\tWarmup: {warmup_steps/total_steps*100:.1f}% | Final LR: 1% of peak")
    
    return scheduler
