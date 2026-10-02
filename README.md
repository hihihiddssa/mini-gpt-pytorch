# mini-gpt-pytorch

基于 PyTorch 搭建并训练的小型字符级 Decoder-only Transformer，覆盖因果多头自注意力、Pre-Norm 残差块、下一字符预测、模型保存加载和自回归生成。

Built and trained a compact character-level Decoder-only Transformer in PyTorch, covering causal multi-head self-attention, residual Pre-Norm blocks, next-token prediction, checkpointing, and autoregressive generation.

项目以可读的模型代码和真实中文小说训练为核心，记录了从基础模型到扩大模型、再到白话长语料训练的三次实验。技术栈：Python、PyTorch、CUDA。

The project combines readable model code with training on real Chinese novels. It records three experiments progressing from a baseline to a larger model and then a longer vernacular corpus. Stack: Python, PyTorch, and CUDA.

## 核心实现 / Core Implementation

- 字符级 tokenizer、随机 batch、右移一位的 next-token 目标 / Character tokenizer, random batches, and next-token targets.
- 可学习的 token 与 position embedding / Learned token and position embeddings.
- 显式实现 causal multi-head self-attention / Explicit causal multi-head self-attention.
- GELU FFN、Pre-Norm、残差连接与堆叠 Decoder blocks / GELU feed-forward networks, Pre-Norm, residual connections, and stacked decoder blocks.
- Cross entropy、反向传播与 AdamW / Cross-entropy loss, backpropagation, and AdamW.
- checkpoint 保存与加载、greedy 和 temperature 采样 / Checkpoint saving/loading, greedy decoding, and temperature sampling.
- 有 CUDA 时自动使用 GPU，否则使用 CPU / Automatic CUDA selection with CPU fallback.

## 当前成果 / Current Results

| 项目 / Item | 结果 / Result |
| --- | --- |
| 模型 / Model | 字符级 Decoder-only Transformer / Character-level decoder-only Transformer |
| 参数 / Parameters | 1,578,804 |
| 语料 / Corpus | 简体《春明外史》，899,430 个字符 / Simplified Chinese Chunming Waishi, 899,430 characters |
| 字符词表 / Vocabulary | 3,764 |
| 上下文 / Context | 128 个字符 / 128 characters |
| 训练 / Training | CUDA，200,000 步 / CUDA, 200,000 steps |
| 最终损失 / Final loss | train: 2.4915; eval: 2.1841 |
| 生成 / Generation | Greedy / temperature sampling，逐字输出 / Character-by-character output |

模型可以延续小说中的词汇、对话形式和部分局部句式；生成样例与实验记录见下文。`eval_loss` 来自训练语料的抽样，不是独立验证集指标。

The model reproduces novel vocabulary, dialogue forms, and some local sentence patterns. Actual samples and experiment records appear below. `eval_loss` is sampled from the training corpus, not a held-out validation metric.

## 实验历程 / Experiment History

### 起点：先跑通完整流程 / Starting Point: An End-to-End Run

最初用小规模自备中文文本，确认“读入文本 → 编码 → batch → 模型 → loss → 参数更新 → 保存 → 生成”能完整运行。随后换成真实小说，观察模型规模、上下文长度和语料对训练及生成的影响。

The project started with a small custom Chinese corpus to verify the complete pipeline: text loading, encoding, batching, modeling, loss, parameter updates, saving, and generation. Later experiments used full novels to explore changes in model size, context length, and corpus.

### 三次小说训练 / Three Novel Training Experiments

| 实验 / Run | 语料 / Corpus | 字符 / Characters | 上下文 / Context | 隐藏维度 / Width | 头 / Heads | 层 / Layers | Dropout | 参数 / Parameters | 步数 / Steps | 最终 train / eval loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A：基础模型 / Baseline | 《孽海花》 / Nie Hai Hua | 254,222 | 32 | 64 | 4 | 2 | 0.0 | 629,883 | 50,000 | 3.7610 / 3.8021 |
| B：扩大模型 / Larger model | 《孽海花》 / Nie Hai Hua | 254,222 | 128 | 128 | 4 | 3 | 0.1 | 1,662,843 | 50,000 | 2.2395 / 1.8148 |
| C：白话长语料 / Vernacular corpus | 《春明外史》 / Chunming Waishi | 899,430 | 128 | 128 | 4 | 3 | 0.1 | 1,578,804 | 200,000 | 2.4915 / 2.1841 |

**A：** 用较小模型训练真实小说。loss 持续下降，生成中出现小说词汇、人物名和对话标点，但句子经常不连贯。

**A:** The smaller model learned novel vocabulary, names, and dialogue punctuation as loss decreased, but generated sentences were often incoherent.

**B：** 在同一语料上将上下文从 32 增至 128、隐藏维度从 64 增至 128、Decoder blocks 从 2 增至 3，并启用 0.1 dropout。训练 50,000 步后，训练文本上的平均 loss 明显降低。

**B:** On the same corpus, context grew from 32 to 128, hidden width from 64 to 128, and decoder depth from 2 to 3, with dropout of 0.1. After 50,000 steps, average loss on the training corpus was substantially lower.

**C：** 为学习白话表达，改用完整简体《春明外史》，从头训练 200,000 步。模型结构沿用 B；字符词表从 4,091 变为 3,764，因此 embedding 和输出层的参数量也随之减少。最后加入逐字输出，让每个新字符生成后立即显示。

**C:** The corpus changed to the complete Simplified Chinese *Chunming Waishi* for vernacular prose, followed by 200,000 steps of training from scratch. Architecture stayed the same as B. Vocabulary decreased from 4,091 to 3,764 characters, reducing embedding and output-layer parameter counts. Character-by-character output was added to display each newly generated character immediately.

### 关键损失记录 / Selected Loss Records

以下来自实际训练终端记录，只展示关键里程碑。实验 C 保留的日志覆盖 162,000–200,000 步。

The following values come from actual training terminal records, with only key milestones shown. The retained log for C covers steps 162,000–200,000.

| 实验 / Run | Step | train_loss | eval_loss |
| --- | ---: | ---: | ---: |
| A | 1 | 8.4570 | 8.4632 |
| A | 5,000 | 5.0525 | 4.9422 |
| A | 20,000 | 4.1591 | 4.2419 |
| A | 50,000 | 3.7610 | 3.8021 |
| B | 1 | 8.4597 | 8.4427 |
| B | 1,000 | 5.3095 | 5.2278 |
| B | 5,000 | 4.5476 | 4.0460 |
| B | 50,000 | 2.2395 | 1.8148 |
| C | 162,000 | 2.4168 | 2.2658 |
| C | 175,000 | 2.4678 | 2.2304 |
| C | 190,000 | 2.5351 | 2.2071 |
| C | 200,000 | 2.4915 | 2.1841 |

`train_loss` 是当前训练 batch 的损失；`eval_loss` 是模型关闭 dropout 后，从**同一份训练文本**随机抽取 20 个 batch 的平均损失，没有独立验证集。两者使用的样本与 dropout 状态不同，所以 eval loss 可以低于 train loss。它们反映对训练语料的拟合程度，不代表对未见文本的泛化能力。

`train_loss` measures the current training batch. `eval_loss` averages 20 random batches from **the same training corpus**, with dropout disabled; there is no separate validation set. Different samples and dropout settings mean eval loss can be lower than train loss. These metrics describe fit to the corpus rather than generalization to unseen text.

A 与 B 的同语料结果可作为本项目内的实验观察，但并非严格控制变量的评测；C 更换了语料及词表，不能凭最终 loss 直接判断它优于或劣于 B。

A and B provide observations on the same corpus, rather than a controlled benchmark. C uses a different corpus and vocabulary, so its final loss cannot directly rank it against B.

## 实际生成样例 / Actual Generated Samples

以下均来自真实终端输出，保留原有错误，不润色。输出包含提示词；生成文本保留中文，不作翻译。随机采样的再次运行结果可能不同。

These samples are taken from actual terminal output, including prompts and original errors. Generated text is kept in Chinese without editing or translation. Random sampling may produce different results on another run.

### 基础模型 A / Baseline A

提示词 / Prompt：`第四回`

```text
第四回请训罢，一坐半晌，白坐在前面上，一个红三绿色，锦吟道：「还要的挚日，哪里是黑旗将军火的呢？我们做得令，没有所问你去刚赴理也深，实在大家未足。这一天，我也决定说的。」一个何安道：「死妓街的主意发？你们不
```

### 当前模型 C / Current Model C

输出开头 / Output beginning：`过了几日`

```text
过了几日云雨了。再要到医院里去诊病，不能久等，也就回来，请医生在报馆里来。到了那些医生送来，一走过来，不四五分钟，满每想过我明天就买主会饭碗一碗饭。”

这时何太太心里早要问，只好忽然一动，一个早到了晚上，他
```

输出开头 / Output beginning：`你好`

```text
你好办了一个军官，你是可以去看看的，可是现在……”

李老太太站在里，手里拿着两只手，叫了一声“回来”，叫我打开保险箱子吗？再一看打扑克，捏住了。当他出去，倒是填不了的，翻出一只手来。口里将门一推，在门口
```

## 快速运行 / Quick Start

需要 Python 3.10+ 和 PyTorch。在项目目录和自己的 Python 环境中运行：

Requires Python 3.10+ and PyTorch. Run from the project directory in your Python environment:

```bash
git clone https://github.com/hihihiddssa/mini-gpt-pytorch.git
cd mini-gpt-pytorch
pip install -r requirements.txt

# 从随机权重训练，默认 200,000 步 / Train from scratch, default 200,000 steps
python train.py

# 默认提示词“陈国英道：”，生成 100 个新字符 / Default prompt: 陈国英道：; 100 new characters
python generate.py

# 指定提示词和长度 / Set prompt and output length
python generate.py --prompt 杨杏园 --max-new-tokens 300 --temperature 0.8

# 贪心生成 / Greedy generation
python generate.py --prompt 杨杏园 --greedy

# 调整逐字显示间隔（秒）/ Adjust delay between characters (seconds)
python generate.py --delay 0.1
python generate.py --delay 0
```

`generate.py` 每生成一个新字符就立即打印并刷新输出，默认额外间隔 0.03 秒；`--delay 0` 取消额外等待，仍然逐字输出。这个参数只改变显示节奏，不改变采样概率。

`generate.py` prints and flushes each newly generated character, with an additional delay of 0.03 seconds by default. `--delay 0` removes the pause while retaining immediate character output. This option changes display timing, not sampling probabilities.

有 CUDA 时训练和生成自动使用 GPU，否则使用 CPU。`--data` 指定其他 UTF-8 训练文件，`--steps` 覆盖训练步数，`--checkpoint` 指定模型路径；默认路径是 `checkpoints/latest.pt`。

Training and generation select CUDA when available and otherwise use CPU. `--data` selects another UTF-8 training file, `--steps` overrides the training step count, and `--checkpoint` sets the model path. The default checkpoint is `checkpoints/latest.pt`.

**每次运行 `train.py` 都从随机权重开始，训练结束才保存，并覆盖所指定的 checkpoint。** 当前保存模型权重、配置、字符表和步数；未保存优化器状态，也没有断点续训入口。生成使用 checkpoint 内的词表及模型配置，提示词不能包含该词表以外的字符。

**Each invocation of `train.py` starts from random weights and saves only at completion, overwriting the selected checkpoint.** It stores weights, configuration, character vocabulary, and step count, but not optimizer state; there is no training-resume interface. Generation uses the checkpoint's vocabulary and architecture, and prompts must contain only characters in that vocabulary.

`checkpoints/` 由 Git 忽略，本仓库不上传训练权重。克隆后需要先训练，或放入与代码匹配的 checkpoint，再运行生成。

`checkpoints/` is ignored by Git, so trained weights are not uploaded. After cloning, train a model or supply a compatible checkpoint before generation.

## 文件与数据流 / Files and Data Flow

| 文件 / File | 用途 / Purpose |
| --- | --- |
| `config.py` | 模型与训练参数 / Model and training settings |
| `tokenizer.py` | 字符与 ID 互转 / Character-to-ID encoding and decoding |
| `dataset.py` | 随机输入及右移目标 / Random inputs and shifted targets |
| `model.py` | Embedding、Attention、FFN、Decoder blocks、GPT、采样 / Model components and sampling |
| `train.py` | Loss、AdamW、训练循环及保存 / Loss, optimizer, training loop, and saving |
| `generate.py` | 加载 checkpoint 并逐字输出 / Checkpoint loading and character-by-character output |
| `data/train.txt` | 当前简体小说语料 / Current Simplified Chinese corpus |
| `data/SOURCE.md` | 来源、处理和统计 / Provenance, processing, and statistics |

`文本 → 字符 ID → x/y（y 右移一位）→ token + position embedding → Decoder blocks → logits → cross entropy → backward → AdamW`

`Text → character IDs → x/y (targets shifted by one character) → token + position embeddings → decoder blocks → logits → cross-entropy loss → backward → AdamW`

生成时，每一步只使用最后一个位置的 logits 选择下一字符，将它追加到输入，再继续预测；超出上下文长度时只保留最近 128 个字符参与计算。

During generation, the last position's logits select the next character, which is appended before the next prediction. When context exceeds the limit, only the latest 128 characters enter the model.

## 当前配置 / Current Configuration

| 参数 / Parameter | 值 / Value | 含义 / Meaning |
| --- | ---: | --- |
| `block_size` | 128 | 上下文字符数 / Context length |
| `n_embd` | 128 | 隐藏维度 / Hidden width |
| `n_head` | 4 | 每头 32 维 / 32 dimensions per head |
| `n_layer` | 3 | Decoder blocks |
| `dropout` | 0.1 | 丢弃概率 / Dropout probability |
| `batch_size` | 16 | 每次更新的片段数 / Sequences per update |
| `max_steps` | 200,000 | 总更新次数 / Total updates |
| `eval_interval` | 250 | 平均 loss 输出间隔 / Loss reporting interval |
| `eval_batches` | 20 | 每次平均 loss 的 batch 数 / Batches per loss estimate |
| `learning_rate` | 0.0003 | AdamW 学习率 / AdamW learning rate |
| `seed` | 42 | 随机种子 / Random seed |

当前模型 **1,578,804 参数**。输入与目标为 `[16,128]`，embedding 为 `[16,128,128]`，注意力权重为 `[16,4,128,128]`，logits 为 `[16,128,3764]`。

The current model has **1,578,804 parameters**. Inputs and targets are `[16,128]`, embeddings `[16,128,128]`, attention weights `[16,4,128,128]`, and logits `[16,128,3764]`.

## 当前语料 / Current Corpus

`data/train.txt` 是张恨水的简体《春明外史》，来自[虚阁网全文](https://www.xuges.com/xdmj/zhs/cmws/index.htm)。保留第 1–86 回正文，移除网页导航、脚本、缩进与序言，用 OpenCC 转为简体，UTF-8 保存；未重复或补写正文。

`data/train.txt` contains all 86 chapters of Zhang Henshui's *Chunming Waishi*, from the linked online text. Navigation, scripts, indentation, and prefaces were removed. OpenCC converted it to Simplified Chinese, saved as UTF-8, with no repetition or invented additions.

| 项目 / Item | 数值 / Value |
| --- | ---: |
| 总字符（含标点、换行）/ Total characters, including punctuation and newlines | 899,430 |
| 汉字 / Han characters | 738,250 |
| 字符词表 / Vocabulary | 3,764 |
| 章节 / Chapters | 86 |

这是一部民国白话小说，仍有少量诗词及旧式表达。约 90 万总字符与纯汉字数采用不同统计口径。来源、预处理、原作版权信息和 SHA-256 见 [data/SOURCE.md](data/SOURCE.md)。旧实验 A/B 使用 [Project Gutenberg 的《孽海花》版本](https://www.gutenberg.org/ebooks/25128)，35 回、4,091 字符词表；当前训练文件已替换为新语料。

This is vernacular Republican-era fiction with some poetry and period expressions. Total character counts and Han-only counts differ. See [data/SOURCE.md](data/SOURCE.md) for provenance, preprocessing, original-work copyright information, and SHA-256. Earlier runs A/B used the linked Project Gutenberg edition of *Nie Hai Hua*, with 35 chapters and a 4,091-character vocabulary. The current training file contains the new corpus.

## 局限与说明 / Limitations and Notes

- 当前生成仍存在人物、动作和逻辑错误，128 字符的上下文也限制了长距离情节的一致性。
- 模型仅在一本小说上进行下一字符训练，没有指令微调，不具备聊天助手的指令遵循能力。
- 没有独立验证集；训练语料抽样 loss 不能作为未见文本上的泛化指标，不同语料的 loss 也不能直接比较。
- checkpoint 用于加载与生成；当前训练脚本只在结束时保存，未保存优化器状态，也未提供断点续训入口。
- 仓库不包含本地训练权重，使用者需先训练或提供匹配的 checkpoint。

- Generation still contains errors in characters, actions, and logic. The 128-character context limits longer story continuity.
- Training predicts the next character in a single novel without instruction tuning; this is not an instruction-following chat assistant.
- There is no held-out validation set. Loss sampled from training text does not measure generalization, and losses on different corpora are not directly comparable.
- Checkpoints support loading and generation. Training saves only at completion, without optimizer state or a resume interface.
- Local trained weights are not included in this repository; train a model or supply a compatible checkpoint.

## 开发说明 / Development Notes

初始项目搭建使用了 AI 辅助，随后围绕代码阅读、配置调整、语料处理、训练调试和生成效果开展学习与实验。本项目使用 “Built and trained” 描述工作，不声称所有代码均独立手写。

AI assistance was used for the initial project setup. Subsequent learning and experiments focused on reading the code, adjusting configurations, preparing corpora, debugging training, and inspecting generation. “Built and trained” describes the work without claiming that every line was independently handwritten.
