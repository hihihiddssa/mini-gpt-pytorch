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
    text = args.data.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    if len(token_ids) <= config.block_size:
        parser.error(f"Training text needs more than {config.block_size} characters")

    model = GPT(config, tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    print(f"device={device} characters={len(text)} vocab={tokenizer.vocab_size} parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    model.train()
    for step in range(1, config.max_steps + 1):
        x, y = make_batch(token_ids, config.batch_size, config.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % config.eval_interval == 0 or step == config.max_steps:
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
