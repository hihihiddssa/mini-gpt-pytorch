import argparse
import math
import time
from pathlib import Path
import torch
from config import Config
from model import GPT
from tokenizer import CharTokenizer
ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description='Generate text with a trained checkpoint')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints' / 'latest.pt')
    parser.add_argument('--prompt', default='陈国英道：')
    parser.add_argument('--max-new-tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--delay', type=float, default=0.03, help='Pause between generated characters in seconds; 0 disables the pause')
    args = parser.parse_args()
    if not args.prompt:
        parser.error('--prompt must contain at least one character')
    if args.max_new_tokens < 0:
        parser.error('--max-new-tokens cannot be negative')
    if not math.isfinite(args.delay) or args.delay < 0:
        parser.error('--delay must be a finite nonnegative number')
    if not args.greedy and args.temperature <= 0:
        parser.error('--temperature must be positive')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = Config(**checkpoint['config'])
    tokenizer = CharTokenizer(checkpoint['chars'])
    model = GPT(config, tokenizer.vocab_size).to(device)
    model.load_state_dict(checkpoint['model'])

    # Generate text
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    print(args.prompt, end='', flush=True)
    for _ in range(args.max_new_tokens):
        input_ids = model.generate(input_ids, 1, args.temperature, args.greedy)
        next_char = tokenizer.decode([input_ids[0, -1].item()])
        print(next_char, end='', flush=True)
        if args.delay > 0:
            time.sleep(args.delay)
    print()


if __name__ == '__main__':
    main()
