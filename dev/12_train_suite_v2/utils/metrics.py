import math
import torch

def compute_loss_perplexity(loss):
    return math.exp(loss) if loss < 10 else float('inf')

def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == labels).float()
    mask = (labels != -100).float()
    return (correct * mask).sum() / mask.sum() if mask.sum() != 0 else 0.0

def tokens_per_second(num_tokens, elapsed_time):
    return (num_tokens / elapsed_time) if elapsed_time > 0 else 0
