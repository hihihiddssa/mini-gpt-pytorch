import torch

def make_batch(token_ids, batch_size, block_size, device):
    if len(token_ids) <= block_size:
        raise ValueError('Training text must contain more than block_size characters')
    starts = torch.randint(0, len(token_ids) - block_size, (batch_size,))#随机生成 batch_size 个起始位置，范围是 [0, len(token_ids) - block_size)，确保每个片段都能完整取到 block_size 个字符
    x = torch.stack([token_ids[i:i + block_size] for i in starts])#把token_ids中从每个起始位置开始的 block_size 个字符取出来，组成一个 batch_size x block_size 的张量 x
    y = torch.stack([token_ids[i + 1:i + block_size + 1] for i in starts])#把 token_ids 中从每个起始位置的下一个字符开始的 block_size 个字符取出来，组成一个 batch_size x block_size 的张量 y，作为 x 的正确答案
    return (x.to(device), y.to(device))
