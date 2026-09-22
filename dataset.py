"""从整篇文本的 token ID 中，随机截取下一字符预测所需的训练样本。"""

import torch


def make_batch(token_ids, batch_size, block_size, device):
    """返回 x/y，形状均为 [B, T]；y 是 x 向右移动一个字符后的答案。"""
    # 例如原文是「春天来了」，block_size=3：
    # x 可以是「春天来」，对应的 y 就是「天来了」。
    # 因为 y 比 x 多用到后面一个字符，所以文本长度必须超过 block_size。
    if len(token_ids) <= block_size:
        raise ValueError("Training text must contain more than block_size characters")

    # 从所有合法起点里随机抽 batch_size 个。不同样本可能重叠，也可能重复。
    # randint 的上界不包含在结果中；最大起点为 len(token_ids)-block_size-1。
    starts = torch.randint(0, len(token_ids) - block_size, (batch_size,))

    # 每个起点截出 block_size 个输入 ID，再用 stack 叠成 [B, T]。
    # 这里 token_ids 在 CPU 上，先在 CPU 切片，最后整批搬到训练设备。
    x = torch.stack([token_ids[i:i + block_size] for i in starts])
    # 目标从起点的下一个字符开始，因此 x[b, t] 要预测 y[b, t]。
    y = torch.stack([token_ids[i + 1:i + block_size + 1] for i in starts])
    # device 通常是「cuda」或「cpu」；模型和数据必须在同一设备上。
    return x.to(device), y.to(device)
