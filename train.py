import argparse
from pathlib import Path

import torch

from config import Config
from dataset import make_batch
from model import GPT
from tokenizer import CharTokenizer


ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def estimate_loss(model, token_ids, config, device):
    model.eval()
    losses = []
    for _ in range(config.eval_batches):
        x, y = make_batch(token_ids, config.batch_size, config.block_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser(description="Train a small character-level GPT")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "train.txt")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "latest.pt")
    parser.add_argument("--steps", type=int, default=None, help="Override config max_steps")
    args = parser.parse_args()

    config = Config()
    if args.steps is not None:
        if args.steps < 1:
            parser.error("--steps must be at least 1")
        config.max_steps = args.steps
    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #加载训练文本并建立 tokenizer
    text = args.data.read_text(encoding="utf-8")#加载训练文本
    tokenizer = CharTokenizer.from_text(text)#建立'字符->ID' 的映射表
    token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)#得到训练文本对应的 ID 列表然后转为张量

    #报错机制
    if len(token_ids) <= config.block_size:
        parser.error(f"Training text needs more than {config.block_size} characters")


    #创建模型和优化器
    model = GPT(config, tokenizer.vocab_size).to(device)#传入一些参数 进入模型 按照顺序创建(tokenembedding, position_embedding, decoder_blocks, layer_norm, lm_head) 并把模型放到 device 上
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)#创建优化器adamW 之后根据这些参数的梯度 根据adamW算法更新GPT的参数
    print(f"device={device} characters={len(text)} vocab={tokenizer.vocab_size} parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)

    #开始训练
    model.train()#告诉pytorch，模型要进入训练模式，启用dropout等
    for step in range(1, config.max_steps + 1):#step从1开始，直到max_steps
        x, y = make_batch(token_ids, config.batch_size, config.block_size, device)#x是输入y是正确答案 一次随机从train.txt中取16(batch_size)个长度为32(block_size)的片段
        _, loss = model(x, y)#return的是logits和loss
        #无需要model.forward等 内部已经帮你弄好了 直接就是model->nn.Module.call->GPT.forward
        optimizer.zero_grad(set_to_none=True)#把上一次的梯度清零 因为pytorch默认是累加梯度的 比如上一次0.2 这一次0.3 那么总梯度就是0.5
        loss.backward()#计算梯度 反向传播 计算每个参数的梯度
        optimizer.step()##根据梯度更新参数
        if step == 1 or step % config.eval_interval == 0 or step == config.max_steps:#每隔eval_interval步 或者是第一步 或者是最后一步 就打印一次loss
            avg_loss = estimate_loss(model, token_ids, config, device)
            print(f"step={step} train_loss={loss.item():.4f} eval_loss={avg_loss:.4f}", flush=True)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": config.to_dict(),
        "chars": tokenizer.chars,
        "step": config.max_steps,
    }, args.checkpoint)
    print(f"checkpoint={args.checkpoint.resolve()}", flush=True)


if __name__ == "__main__":
    main()
