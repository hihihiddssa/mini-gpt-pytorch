"""读取已训练的 checkpoint，从给定 prompt 开始逐字符生成文字。"""

import argparse
from pathlib import Path

import torch

from config import Config
from model import GPT
from tokenizer import CharTokenizer


# 用脚本所在目录作为默认路径的起点，避免受当前终端目录影响。
ROOT = Path(__file__).resolve().parent


def main():
    """解析生成参数，重建模型，载入权重，然后生成并打印文字。"""
    # argparse 把命令行里的 --prompt、--greedy 等选项变成 args 的属性。
    parser = argparse.ArgumentParser(description="Generate text with a trained checkpoint")
    # 默认读取本项目训练产生的 checkpoints/latest.pt。
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "latest.pt")
    # prompt 是生成的开头；它本身也会包含在最终输出里。
    parser.add_argument("--prompt", default="面包")
    # 生成多少个新字符；例如 100 不是最终总长度，因为还要加上 prompt。
    parser.add_argument("--max-new-tokens", type=int, default=100)
    # 仅在随机采样时使用温度。小于 1 更保守，大于 1 更随机。
    parser.add_argument("--temperature", type=float, default=1.0)
    # store_true 表示写 --greedy 时为 True，不写时为 False。
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()
    # 至少要有一个字符，模型才能用最后一个位置预测后续字符。
    if not args.prompt:
        parser.error("--prompt must contain at least one character")
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens cannot be negative")

    # 加载到可用的 GPU 或 CPU；模型与输入张量必须在同一设备上。
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # checkpoint 是 train.py 保存的字典：模型权重、配置、字符表和步数。
    # map_location 避免在无 GPU 的机器上加载 GPU 保存的权重时出错。
    # weights_only=False 只用于读取本项目自己生成、可信任的 checkpoint；
    # 不应对来源不明的 .pt 文件这样加载。
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # ** 会把保存的参数字典展开，例如 block_size=32、n_embd=64。
    # 配置和词表必须与训练时相同，否则权重张量的形状可能对不上。
    config = Config(**checkpoint["config"])
    tokenizer = CharTokenizer(checkpoint["chars"])
    # 先用同样的结构创建模型，再用 state_dict 填入训练好的参数。
    model = GPT(config, tokenizer.vocab_size).to(device)
    model.load_state_dict(checkpoint["model"])
    # encode 把 prompt 字符变成 ID。外层 [...] 增加 batch 维度 B=1。
    # dtype=torch.long 是 nn.Embedding 所需的整数类型；形状为 [1,T]。
    # 如果 prompt 包含训练词表里没有的字符，encode 会报错。
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    # 模型每次生成一个 ID，循环 max_new_tokens 次；返回值也包含原 prompt。
    output_ids = model.generate(input_ids, args.max_new_tokens, args.temperature, args.greedy)
    # 只有一条样本，所以取 output_ids[0]；再把 ID 还原成可读字符串。
    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    # 直接运行 python generate.py 时执行；被其他文件导入时不执行。
    main()
