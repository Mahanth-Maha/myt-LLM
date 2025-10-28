import torch


def collate_fn(batch):
    if not batch:
        raise ValueError("Empty batch")

    sequences = torch.stack(batch, dim=0)

    input_ids = sequences[:, :-1]
    labels = sequences[:, 1:]

    return {
        'input_ids': input_ids,
        'labels': labels
    }


def debug_collate_fn(batch):
    result = collate_fn(batch)

    result.update({
        'batch_size': len(batch),
        'sequence_length': batch[0].shape[0] - 1 if batch else 0,
        'total_tokens': sum(seq.shape[0] - 1 for seq in batch),
        'vocab_range': {
            'min': result['input_ids'].min().item(),
            'max': result['input_ids'].max().item()
        }
    })

    return result


def adaptive_collate_fn(batch, pad_token_id=0):
    if not batch:
        raise ValueError("Empty batch")

    max_len = max(seq.shape[0] for seq in batch) - 1

    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_len),
                           pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, seq in enumerate(batch):
        seq_len = seq.shape[0] - 1
        input_ids[i, :seq_len] = seq[:-1]
        labels[i, :seq_len] = seq[1:]
        attention_mask[i, :seq_len] = True

    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask
    }
