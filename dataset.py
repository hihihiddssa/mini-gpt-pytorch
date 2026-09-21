import torch


def make_batch(token_ids, batch_size, block_size, device):
    """Sample next-character examples; x and y are shifted by one position."""
    if len(token_ids) <= block_size:
        raise ValueError("Training text must contain more than block_size characters")
    starts = torch.randint(0, len(token_ids) - block_size, (batch_size,))
    x = torch.stack([token_ids[i:i + block_size] for i in starts])
    y = torch.stack([token_ids[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)
