# Mini GPT From Scratch

一个可以在真实文本上训练、保存并生成文字的最小 Decoder-only Transformer。代码使用 PyTorch 基础组件实现注意力和模型结构，适合从数据流开始逐模块学习。训练语料很小，生成质量只用于验证流程，不能代表实用语言模型。

## 运行

需要 Python 3.10+ 和 PyTorch。可在自己的 Python 环境中安装：

```bash
pip install -r requirements.txt
python train.py --steps 50
python generate.py --prompt 春天 --max-new-tokens 60 --greedy
python generate.py --prompt 春天 --max-new-tokens 60 --temperature 0.8
```

不传 `--steps` 时使用 `config.py` 中的 300 步，适合先验证训练流程。当前附带的 checkpoint 用扩充后的语料训练了 1200 步；若想重新得到接近的结果，运行 `python train.py --steps 1200`。程序自动选择 CUDA，未检测到 CUDA 时使用 CPU。训练结果保存在 `checkpoints/latest.pt`，生成程序会从这里加载。`--data` 和 `--checkpoint` 可以指定其他路径。重新训练会覆盖这个 checkpoint。

字符表从训练文本自动构建并与模型一起保存。生成提示词中的每个字必须在训练文本中出现过；如果换语料，请重新训练。默认语料是一篇约 2800 字的原创短篇，只供小规模学习。训练脚本显示的 `eval_loss` 是从同一份训练文本随机取样得到的平均 loss，不是独立验证集指标。

## 文件

| 文件 | 用途 |
| --- | --- |
| `data/train.txt` | 最小真实文本语料 |
| `tokenizer.py` | 字符与 token ID 互转 |
| `dataset.py` | 随机抽取输入和右移一位的目标 |
| `model.py` | Embedding、因果多头注意力、FFN、Pre-Norm Decoder blocks、GPT 和生成 |
| `config.py` | 模型和训练参数 |
| `train.py` | AdamW 训练、loss 输出和 checkpoint 保存 |
| `generate.py` | checkpoint 加载、贪心或 temperature 采样 |

## 一批数据的路径

`文本 → 字符 ID → x/y（y 右移一位）→ token + position embedding → Decoder blocks → logits → cross entropy → 反向传播 → AdamW`。

关键 shape（默认 `B=16, T=32, C=64, H=4`）：`x/y [B,T]`，embedding `[B,T,C]`，每组 Q/K/V `[B,H,T,C/H]`，注意力权重 `[B,H,T,T]`，logits `[B,T,V]`。

## 后续学习顺序

1. 先运行训练和生成，确认完整流程。
2. 追踪一个 batch 从原始文字到 loss 的完整数据流。
3. 依次学习 Embedding → Self-Attention → Multi-Head → FFN → Norm/Residual → Decoder Block → 完整模型。
4. 学习 next-token loss、反向传播、AdamW 和 checkpoint。
5. 学习 greedy 和 temperature generation。
6. 最后形成自己的注解版代码。
