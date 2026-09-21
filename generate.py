import argparse
from pathlib import Path

import torch

from config import Config
from model import GPT
from tokenizer import CharTokenizer


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "latest.pt")
    parser.add_argument("--prompt", default="面包")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()
    if not args.prompt:
        parser.error("--prompt must contain at least one character")
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens cannot be negative")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = Config(**checkpoint["config"])
    tokenizer = CharTokenizer(checkpoint["chars"])
    model = GPT(config, tokenizer.vocab_size).to(device)
    model.load_state_dict(checkpoint["model"])
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    output_ids = model.generate(input_ids, args.max_new_tokens, args.temperature, args.greedy)
    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
