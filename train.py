"""读取真实文本，训练下一字符预测模型，并把训练结果保存为 checkpoint。"""

import argparse
from pathlib import Path

import torch

from config import Config
from dataset import make_batch
from model import GPT
from tokenizer import CharTokenizer


# __file__ 是当前脚本路径；resolve() 得到绝对路径，parent 是项目目录。
# 因此从任何工作目录运行本脚本，默认文件路径都仍指向本项目。
ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def estimate_loss(model, token_ids, config, device):
    """从训练文本随机取若干 batch，计算它们的平均 loss。"""
    # 这里没有独立验证集；函数名中的 eval 仅表示「暂时不更新参数」。
    # no_grad() 不记录梯度，既节省内存，也避免评估影响后续反向传播。
    # eval() 关闭 Dropout 等训练专用行为。
    model.eval()
    losses = []
    for _ in range(config.eval_batches):
        # 每轮重新随机抽一批，因此平均值比单一 batch 的 loss 更平稳。
        x, y = make_batch(token_ids, config.batch_size, config.block_size, device)
        _, loss = model(x, y)
        # .item() 把只有一个数的 tensor 转成普通 Python 浮点数。
        losses.append(loss.item())
    # 评估结束后切回训练模式，后面的训练循环才能正常运行。
    model.train()
    return sum(losses) / len(losses)


def main():
    """按「参数 → 数据 → 模型 → 训练 → 保存」的顺序执行一次训练。"""
    # argparse 读取命令行参数。例如 python train.py --steps 100。
    # 若没有提供某个参数，就使用下面给出的 default 值。
    parser = argparse.ArgumentParser(description="Train a small character-level GPT")
    # 你原先给 --data 写的注释是对的：它指定训练文本路径。
    # type=Path 会把输入路径转为 Path 对象；默认值为本项目的 data/train.txt。
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "train.txt")
    # --checkpoint 指定模型保存位置；目录不存在时会在最后创建。
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "latest.pt")
    # --steps 可临时覆盖 config.py 的默认训练步数，不必改源码。
    parser.add_argument("--steps", type=int, default=None, help="Override config max_steps")
    args = parser.parse_args()

    # Config() 创建参数对象。只有用户明确给了 --steps，才覆盖 max_steps。
    config = Config()
    if args.steps is not None:
        if args.steps < 1:
            # parser.error 会显示友好的命令行错误并停止程序。
            parser.error("--steps must be at least 1")
        config.max_steps = args.steps
    # 固定 PyTorch 随机种子，影响参数初始化和随机 batch 抽样。
    torch.manual_seed(config.seed)
    # 有可用的 NVIDIA CUDA 时用 GPU，否则自动在 CPU 上运行。
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 从 args.data 指定的路径读取整篇原文。read_text() 返回 Python 字符串，
    # 自动关闭文件；encoding="utf-8" 保证中文按 UTF-8 解码。
    text = args.data.read_text(encoding="utf-8")
    # 这里修正你原先注释中的一个细节：from_text(text) 创建 tokenizer 对象
    # 和字符表，还没有生成数字列表；真正「文字 → ID」发生在下一行的 encode。
    tokenizer = CharTokenizer.from_text(text)
    # encode(text) 把每个字符变成 ID；torch.tensor 把 ID 列表变为张量。
    # nn.Embedding 需要整数索引，因此使用 torch.long（64 位整数）。
    # 此时 token_ids 是一维张量 [整篇文本的字符数]，仍存放在 CPU。
    token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # 截出长度 T 的输入时，目标 y 还需要后面一个字符。
    if len(token_ids) <= config.block_size:
        parser.error(f"Training text needs more than {config.block_size} characters")

    # 依据配置和词表大小创建新模型，并把参数搬到 device。
    # 这里是从随机权重开始训练，不会自动加载旧 checkpoint。
    model = GPT(config, tokenizer.vocab_size).to(device)
    # AdamW 持有所有可训练参数，负责根据梯度更新它们。
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    # 参数量是各个张量元素数之和；它不是语料字符数。
    print(f"device={device} characters={len(text)} vocab={tokenizer.vocab_size} parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    model.train()
    # 一个 step = 抽一批数据、计算 loss、反向传播、更新一次模型参数。
    for step in range(1, config.max_steps + 1):
        # x、y 均为 [B,T]；y[b,t] 是 x[b,t] 后面的那个字符。
        x, y = make_batch(token_ids, config.batch_size, config.block_size, device)
        # 前向传播得到预测分数 [B,T,V] 和标量 loss；此处不需直接用 logits。
        _, loss = model(x, y)
        # 清除上一步的梯度，否则 PyTorch 默认会把新梯度累加到旧梯度上。
        optimizer.zero_grad(set_to_none=True)
        # 从 loss 沿计算图反向求导，把梯度写入每个参数的 .grad。
        loss.backward()
        # AdamW 读取这些梯度并更新参数；真正「学到东西」发生在这里。
        optimizer.step()
        if step == 1 or step % config.eval_interval == 0 or step == config.max_steps:
            # 当前 batch 的 train_loss 会波动；多批平均的 eval_loss 更平稳。
            # 两者都使用同一份训练文本，不能把 eval_loss 当成泛化能力指标。
            avg_loss = estimate_loss(model, token_ids, config, device)
            print(f"step={step} train_loss={loss.item():.4f} eval_loss={avg_loss:.4f}", flush=True)

    # parents=True 表示递归创建父目录；exist_ok=True 表示目录已有也不报错。
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    # checkpoint 是一个字典：保存权重、重建模型所需配置、字符表和步数。
    # 只保存权重，没有保存 AdamW 的内部状态；它可用于生成或重新加载模型，
    # 但不是可无缝续训的完整训练状态。
    torch.save({
        "model": model.state_dict(),
        "config": config.to_dict(),
        "chars": tokenizer.chars,
        "step": config.max_steps,
    }, args.checkpoint)
    print(f"checkpoint={args.checkpoint.resolve()}", flush=True)


if __name__ == "__main__":
    # 直接运行 python train.py 时执行 main()；被其他文件 import 时不执行。
    main()
