# Mini GPT From Scratch

一个能从真实中文文本学习下一字符预测、保存模型并生成文字的小型 Decoder-only Transformer。使用 PyTorch 基础组件实现因果多头注意力、FFN、Pre-Norm 残差 Decoder blocks 和完整 GPT 模型，适合逐步学习数据流与训练机制。

当前语料是简体版《孽海花》，已完成一次本机 CUDA 50,000 步训练。模型能够生成带有小说词汇和对话形式的文字，但句子仍有明显的语义错误，不能视为成熟的文本生成模型。

## 运行

需要 Python 3.10+ 和 PyTorch。在项目目录和自己的 Python 环境中运行：

```bash
pip install -r requirements.txt

# 快速检查训练流程，单独保存以保留已有的 latest.pt
python train.py --steps 50 --checkpoint checkpoints/smoke.pt

# 按当前默认配置训练 50,000 步
python train.py

# 从当前 checkpoint 生成，默认 prompt 为“第四回”
python generate.py
python generate.py --prompt 第四回 --max-new-tokens 100 --temperature 1.0
python generate.py --prompt 第四回 --max-new-tokens 100 --greedy
```

程序自动使用可用的 CUDA，否则使用 CPU。`--data` 可以指定其他 UTF-8 文本，`--checkpoint` 可以指定保存或加载路径。每次运行 `train.py` 都从随机权重开始训练，默认会覆盖 `checkpoints/latest.pt`。

checkpoint 保存模型权重、配置、字符表和训练步数，可用于加载模型和生成；没有保存优化器状态，也没有实现断点续训入口。`checkpoints/` 已被 Git 忽略，GitHub 仓库不包含训练权重。从仓库克隆后，需要先训练，或自行放入相匹配的 checkpoint。

字符表从训练文本构建并写入 checkpoint。提示词中的字符必须出现在该 checkpoint 的词表中。替换语料后应重新训练；生成时使用 checkpoint 中保存的词表与结构配置。

## 当前训练语料

`data/train.txt` 是曾朴的《孽海花》，来源为 [Project Gutenberg，电子书 25128](https://www.gutenberg.org/ebooks/25128)。使用该版本全部 35 回，经 OpenCC 繁转简，并清理网站说明、排版空格和段落内的硬换行。

| 项目 | 数值 |
| --- | --- |
| 正文汉字数 | 216,777 |
| 字符数（含标点、换行等） | 254,222 |
| 字符词表大小 | 4,091 |
| 章节数 | 35 回 |
| 文件编码 | UTF-8，简体中文 |

来源、处理过程与统计见 [data/SOURCE.md](data/SOURCE.md)，原始下载文本的许可页脚见 [data/LICENSE-gutenberg.txt](data/LICENSE-gutenberg.txt)。语料来源提供的版本与其他出版版本可能有差异。

## 当前配置

| 参数 | 数值 | 含义 |
| --- | --- | --- |
| `block_size` | 32 | 每次可见的上下文长度 |
| `n_embd` | 64 | 每个字符的隐藏向量维度 |
| `n_head` | 4 | 注意力头数，每头 16 维 |
| `n_layer` | 2 | Decoder block 层数 |
| `dropout` | 0.0 | 当前不丢弃输出 |
| `batch_size` | 16 | 每次更新使用的文本片段数 |
| `max_steps` | 50,000 | 默认参数更新次数 |
| `eval_interval` | 250 | 平均 loss 的输出间隔 |
| `eval_batches` | 20 | 每次估计平均 loss 的 batch 数 |
| `learning_rate` | 0.0003 | AdamW 学习率 |
| `seed` | 42 | 随机种子 |

本次记录中的模型参数量为 **629,883**。每个训练 batch 的输入和目标形状都是 `[16, 32]`，logits 形状是 `[16, 32, 4091]`。

## 本机训练记录

以下记录来自用户提供的真实终端输出，执行 `python train.py`，使用 CUDA 训练 50,000 步，并保存为 `checkpoints/latest.pt`。

`train_loss` 是当前训练 batch 的损失；`eval_loss` 是从同一份训练文本随机抽取 20 个 batch 得到的平均损失。当前没有独立验证集，不能用这些数字判断对新文本的泛化能力。抽样 loss 会波动，不要求每次打印都下降。

| step | train_loss | eval_loss |
| --- | --- | --- |
| 1 | 8.4570 | 8.4632 |
| 1,000 | 5.6489 | 5.8750 |
| 5,000 | 5.0525 | 4.9422 |
| 10,000 | 4.5305 | 4.5837 |
| 20,000 | 4.1591 | 4.2419 |
| 30,000 | 3.9562 | 4.0618 |
| 40,000 | 3.9804 | 3.9121 |
| 50,000 | 3.7610 | 3.8021 |

<details>
<summary>展开完整训练日志（201 次 loss 输出）</summary>

```text
device=cuda characters=254222 vocab=4091 parameters=629,883
step=1 train_loss=8.4570 eval_loss=8.4632
step=250 train_loss=6.1606 eval_loss=6.3212
step=500 train_loss=6.2958 eval_loss=6.2277
step=750 train_loss=6.2324 eval_loss=6.0217
step=1000 train_loss=5.6489 eval_loss=5.8750
step=1250 train_loss=5.7708 eval_loss=5.7822
step=1500 train_loss=5.6425 eval_loss=5.6690
step=1750 train_loss=5.3924 eval_loss=5.6076
step=2000 train_loss=5.4027 eval_loss=5.5283
step=2250 train_loss=5.4121 eval_loss=5.4302
step=2500 train_loss=5.6412 eval_loss=5.4193
step=2750 train_loss=5.2930 eval_loss=5.3841
step=3000 train_loss=5.1288 eval_loss=5.2411
step=3250 train_loss=5.2908 eval_loss=5.2607
step=3500 train_loss=4.9659 eval_loss=5.1593
step=3750 train_loss=4.8411 eval_loss=5.1168
step=4000 train_loss=5.2359 eval_loss=5.0725
step=4250 train_loss=4.9500 eval_loss=5.0944
step=4500 train_loss=4.8712 eval_loss=5.0743
step=4750 train_loss=5.0325 eval_loss=5.0096
step=5000 train_loss=5.0525 eval_loss=4.9422
step=5250 train_loss=5.0887 eval_loss=4.9745
step=5500 train_loss=5.2152 eval_loss=4.8870
step=5750 train_loss=4.9759 eval_loss=4.9399
step=6000 train_loss=5.0434 eval_loss=4.7862
step=6250 train_loss=4.8934 eval_loss=4.9124
step=6500 train_loss=4.6499 eval_loss=4.8394
step=6750 train_loss=5.0197 eval_loss=4.7750
step=7000 train_loss=4.7808 eval_loss=4.7876
step=7250 train_loss=4.6194 eval_loss=4.7472
step=7500 train_loss=4.8013 eval_loss=4.7518
step=7750 train_loss=4.7728 eval_loss=4.7005
step=8000 train_loss=4.4971 eval_loss=4.7427
step=8250 train_loss=4.7136 eval_loss=4.7199
step=8500 train_loss=4.5423 eval_loss=4.6993
step=8750 train_loss=4.7626 eval_loss=4.6867
step=9000 train_loss=5.0419 eval_loss=4.6524
step=9250 train_loss=4.3186 eval_loss=4.5349
step=9500 train_loss=4.5131 eval_loss=4.6734
step=9750 train_loss=4.7173 eval_loss=4.6138
step=10000 train_loss=4.5305 eval_loss=4.5837
step=10250 train_loss=4.6513 eval_loss=4.5765
step=10500 train_loss=4.8558 eval_loss=4.6432
step=10750 train_loss=4.5252 eval_loss=4.6088
step=11000 train_loss=4.4684 eval_loss=4.5387
step=11250 train_loss=4.6573 eval_loss=4.5540
step=11500 train_loss=4.7411 eval_loss=4.5250
step=11750 train_loss=4.4195 eval_loss=4.5647
step=12000 train_loss=4.5202 eval_loss=4.4884
step=12250 train_loss=4.2189 eval_loss=4.4765
step=12500 train_loss=4.5337 eval_loss=4.4490
step=12750 train_loss=4.3196 eval_loss=4.4314
step=13000 train_loss=4.3029 eval_loss=4.4418
step=13250 train_loss=4.4954 eval_loss=4.4310
step=13500 train_loss=4.1387 eval_loss=4.4173
step=13750 train_loss=4.3897 eval_loss=4.4591
step=14000 train_loss=4.4987 eval_loss=4.3505
step=14250 train_loss=4.6306 eval_loss=4.3492
step=14500 train_loss=4.6628 eval_loss=4.4583
step=14750 train_loss=4.5514 eval_loss=4.4097
step=15000 train_loss=4.4332 eval_loss=4.3873
step=15250 train_loss=4.3789 eval_loss=4.3655
step=15500 train_loss=4.2478 eval_loss=4.4328
step=15750 train_loss=4.5200 eval_loss=4.4235
step=16000 train_loss=4.6039 eval_loss=4.3177
step=16250 train_loss=4.3562 eval_loss=4.3520
step=16500 train_loss=4.3266 eval_loss=4.3111
step=16750 train_loss=4.3863 eval_loss=4.3691
step=17000 train_loss=4.4205 eval_loss=4.2689
step=17250 train_loss=4.3949 eval_loss=4.3148
step=17500 train_loss=4.1528 eval_loss=4.2785
step=17750 train_loss=4.3692 eval_loss=4.2721
step=18000 train_loss=4.0816 eval_loss=4.3623
step=18250 train_loss=4.2626 eval_loss=4.2787
step=18500 train_loss=4.4358 eval_loss=4.2312
step=18750 train_loss=4.0598 eval_loss=4.3240
step=19000 train_loss=4.4618 eval_loss=4.2514
step=19250 train_loss=4.2967 eval_loss=4.2593
step=19500 train_loss=4.0440 eval_loss=4.2354
step=19750 train_loss=4.2732 eval_loss=4.2342
step=20000 train_loss=4.1591 eval_loss=4.2419
step=20250 train_loss=4.2840 eval_loss=4.2697
step=20500 train_loss=4.0983 eval_loss=4.2332
step=20750 train_loss=4.2407 eval_loss=4.2063
step=21000 train_loss=4.2523 eval_loss=4.2611
step=21250 train_loss=3.9012 eval_loss=4.2371
step=21500 train_loss=4.0790 eval_loss=4.2794
step=21750 train_loss=4.0636 eval_loss=4.2172
step=22000 train_loss=4.2968 eval_loss=4.1709
step=22250 train_loss=4.2722 eval_loss=4.2027
step=22500 train_loss=4.0920 eval_loss=4.2098
step=22750 train_loss=4.1484 eval_loss=4.1702
step=23000 train_loss=4.0535 eval_loss=4.2247
step=23250 train_loss=4.1880 eval_loss=4.1810
step=23500 train_loss=3.9265 eval_loss=4.1803
step=23750 train_loss=4.2172 eval_loss=4.1895
step=24000 train_loss=4.0370 eval_loss=4.0885
step=24250 train_loss=4.1062 eval_loss=4.1284
step=24500 train_loss=4.2197 eval_loss=4.0916
step=24750 train_loss=4.1961 eval_loss=4.0984
step=25000 train_loss=4.1020 eval_loss=4.1642
step=25250 train_loss=4.3331 eval_loss=4.1586
step=25500 train_loss=4.2102 eval_loss=4.1571
step=25750 train_loss=4.2569 eval_loss=4.1536
step=26000 train_loss=4.1402 eval_loss=4.0605
step=26250 train_loss=4.2093 eval_loss=4.0930
step=26500 train_loss=3.9338 eval_loss=4.0615
step=26750 train_loss=4.1066 eval_loss=4.1311
step=27000 train_loss=3.8246 eval_loss=4.1098
step=27250 train_loss=4.1873 eval_loss=4.0235
step=27500 train_loss=4.2530 eval_loss=4.0403
step=27750 train_loss=4.1121 eval_loss=4.0917
step=28000 train_loss=3.8069 eval_loss=4.0508
step=28250 train_loss=4.1067 eval_loss=4.1100
step=28500 train_loss=4.2340 eval_loss=4.0027
step=28750 train_loss=3.7413 eval_loss=4.0559
step=29000 train_loss=3.9977 eval_loss=4.0419
step=29250 train_loss=4.0182 eval_loss=4.0897
step=29500 train_loss=4.1050 eval_loss=4.0783
step=29750 train_loss=4.2055 eval_loss=4.0851
step=30000 train_loss=3.9562 eval_loss=4.0618
step=30250 train_loss=3.8064 eval_loss=4.0744
step=30500 train_loss=3.9975 eval_loss=4.0332
step=30750 train_loss=4.0126 eval_loss=4.0248
step=31000 train_loss=4.1599 eval_loss=4.0370
step=31250 train_loss=4.0336 eval_loss=4.0042
step=31500 train_loss=3.8763 eval_loss=4.0076
step=31750 train_loss=3.8954 eval_loss=3.9932
step=32000 train_loss=3.8867 eval_loss=3.9394
step=32250 train_loss=3.9177 eval_loss=4.0711
step=32500 train_loss=4.2831 eval_loss=4.0116
step=32750 train_loss=4.1230 eval_loss=3.9905
step=33000 train_loss=4.0689 eval_loss=4.0414
step=33250 train_loss=3.7700 eval_loss=3.9810
step=33500 train_loss=4.1762 eval_loss=4.0266
step=33750 train_loss=4.0812 eval_loss=3.9666
step=34000 train_loss=4.0103 eval_loss=3.9482
step=34250 train_loss=4.0854 eval_loss=4.0223
step=34500 train_loss=3.9075 eval_loss=4.0098
step=34750 train_loss=4.0725 eval_loss=3.9209
step=35000 train_loss=4.0804 eval_loss=3.9831
step=35250 train_loss=4.1724 eval_loss=3.9551
step=35500 train_loss=3.9746 eval_loss=3.9467
step=35750 train_loss=3.9673 eval_loss=3.9973
step=36000 train_loss=4.1374 eval_loss=3.9123
step=36250 train_loss=3.6808 eval_loss=3.9490
step=36500 train_loss=3.9104 eval_loss=3.9458
step=36750 train_loss=3.7279 eval_loss=3.9993
step=37000 train_loss=4.2736 eval_loss=3.9519
step=37250 train_loss=3.9015 eval_loss=3.9735
step=37500 train_loss=3.8834 eval_loss=3.9986
step=37750 train_loss=3.7406 eval_loss=3.9069
step=38000 train_loss=3.9844 eval_loss=3.9477
step=38250 train_loss=3.8575 eval_loss=3.9250
step=38500 train_loss=3.9957 eval_loss=3.8740
step=38750 train_loss=3.8169 eval_loss=3.9218
step=39000 train_loss=3.9696 eval_loss=3.9255
step=39250 train_loss=4.0914 eval_loss=3.9304
step=39500 train_loss=4.0042 eval_loss=3.8941
step=39750 train_loss=3.8626 eval_loss=3.9074
step=40000 train_loss=3.9804 eval_loss=3.9121
step=40250 train_loss=3.7502 eval_loss=3.9256
step=40500 train_loss=3.8769 eval_loss=3.9049
step=40750 train_loss=3.9129 eval_loss=3.8718
step=41000 train_loss=4.0074 eval_loss=3.7992
step=41250 train_loss=3.8658 eval_loss=3.9151
step=41500 train_loss=3.9830 eval_loss=3.8776
step=41750 train_loss=4.0780 eval_loss=3.9163
step=42000 train_loss=3.9090 eval_loss=3.8379
step=42250 train_loss=4.0298 eval_loss=3.8512
step=42500 train_loss=3.7669 eval_loss=3.9424
step=42750 train_loss=3.7348 eval_loss=3.9201
step=43000 train_loss=4.0921 eval_loss=3.9123
step=43250 train_loss=3.7958 eval_loss=3.8604
step=43500 train_loss=3.8800 eval_loss=3.8956
step=43750 train_loss=3.6784 eval_loss=3.8941
step=44000 train_loss=3.7776 eval_loss=3.8647
step=44250 train_loss=3.9645 eval_loss=3.8445
step=44500 train_loss=3.6928 eval_loss=3.8353
step=44750 train_loss=3.9790 eval_loss=3.8830
step=45000 train_loss=3.7655 eval_loss=3.8684
step=45250 train_loss=3.7776 eval_loss=3.8405
step=45500 train_loss=4.1679 eval_loss=3.7713
step=45750 train_loss=3.4515 eval_loss=3.8551
step=46000 train_loss=3.9146 eval_loss=3.8775
step=46250 train_loss=3.8387 eval_loss=3.8841
step=46500 train_loss=4.0137 eval_loss=3.8047
step=46750 train_loss=3.7707 eval_loss=3.7944
step=47000 train_loss=3.7633 eval_loss=3.8452
step=47250 train_loss=3.8383 eval_loss=3.7837
step=47500 train_loss=3.7867 eval_loss=3.8798
step=47750 train_loss=3.7197 eval_loss=3.7991
step=48000 train_loss=3.8548 eval_loss=3.8361
step=48250 train_loss=3.7890 eval_loss=3.8126
step=48500 train_loss=4.0613 eval_loss=3.8169
step=48750 train_loss=3.7657 eval_loss=3.8136
step=49000 train_loss=3.8963 eval_loss=3.8313
step=49250 train_loss=3.9444 eval_loss=3.8001
step=49500 train_loss=3.8139 eval_loss=3.8291
step=49750 train_loss=3.9549 eval_loss=3.7725
step=50000 train_loss=3.7610 eval_loss=3.8021
checkpoint=checkpoints/latest.pt
```

checkpoint 路径在此使用项目相对路径展示，其余训练数值与原始记录一致。

</details>

## 实际生成结果

以下两次输出摘自同一份终端记录，提示词为 `第四回`。当时直接运行 `generate.py`，使用默认 temperature 1.0 随机采样、生成 100 个新字符；输出包含原提示词。随机采样未设置固定种子，因此再次运行不保证逐字相同。这些示例保留实际错误，没有润色。

第一次：

```text
第四回请训罢，一坐半晌，白坐在前面上，一个红三绿色，锦吟道：「还要的挚日，哪里是黑旗将军火的呢？我们做得令，没有所问你去刚赴理也深，实在大家未足。这一天，我也决定说的。」一个何安道：「死妓街的主意发？你们不
```

第二次：

```text
第四回南洋的猛，老爷午后没起来，叫那车门生。」雯青因道：「别碎乐呀！」美菽素里头问道：「克先请俄、德国天消息下诸位分子。」小燕道：「照这班广东图事业，目瞪起酒醒来。」小燕道：「何听他空著，𫉁云也里只剩她的老
```

可以观察到小说人物名、对话标点和部分句式，但上下文与含义仍不稳定。当前上下文只有 32 个字符，模型规模也很小，loss 下降不等于已经学会写连贯故事。

## 文件

| 文件 | 用途 |
| --- | --- |
| `data/train.txt` | 简体《孽海花》训练语料 |
| `data/SOURCE.md` | 语料来源、处理方法与统计 |
| `data/LICENSE-gutenberg.txt` | 原始下载文本的许可页脚 |
| `tokenizer.py` | 字符与 token ID 互转 |
| `dataset.py` | 随机截取输入和右移一位的预测目标 |
| `model.py` | Embedding、因果多头注意力、FFN、Pre-Norm Decoder blocks、GPT 和生成 |
| `config.py` | 模型结构与训练参数 |
| `train.py` | Cross entropy、AdamW 训练、loss 输出和 checkpoint 保存 |
| `generate.py` | checkpoint 加载、贪心或 temperature 采样 |

## 一批数据的路径

`文本 → 字符 ID → x/y（目标 y 右移一位）→ token + position embedding → Decoder blocks → logits → cross entropy → 反向传播 → AdamW`。

用 `B` 表示 batch size、`T` 表示序列长度、`C` 表示隐藏维度、`H` 表示头数、`V` 表示词表大小：输入与目标 `[B,T]`，embedding `[B,T,C]`，Q/K/V 拆头后 `[B,H,T,C/H]`，注意力权重 `[B,H,T,T]`，logits `[B,T,V]`。生成时，每一步只取最后一个位置的 logits 选择下一个字符，再把它接入上下文。

## 学习顺序

1. 先运行训练和生成，确认完整流程。
2. 追踪一个 batch 从原始文字到 loss 的完整数据流。
3. 依次学习 Embedding → Self-Attention → Multi-Head → FFN → Norm/Residual → Decoder Block → 完整模型。
4. 学习 next-token loss、反向传播、AdamW 和 checkpoint。
5. 学习 greedy 和 temperature generation。
6. 最后形成自己的注解版代码。
