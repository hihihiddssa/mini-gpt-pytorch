"""完整的 Decoder-only Transformer：输入字符 ID，预测每个位置的下一个字符。"""

import torch
from torch import nn
from torch.nn import functional as F

# 本文件常用的形状记号：
# B = batch size，一批有多少段文字；T = 每段有多少个字符；
# C = n_embd，每个字符的内部向量维度；H = n_head，注意力头数；
# D = C // H，每个头负责的维度；V = vocab_size，词表大小。
# 默认训练参数是 B=16、T=32、C=64、H=4、D=16；V 由训练文本决定。


class CausalSelfAttention(nn.Module):
    """让每个位置参考自己及之前的位置，不读取未来的正确答案。"""

    def __init__(self, config):
        super().__init__()
        # C 必须能平分给 H 个头；否则每个头无法分到同样多的维度。
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head

        # 一次线性变换同时计算 Query、Key、Value，输出最后一维为 3*C。
        # Q 可以理解为当前位置想找什么，K 是各位置提供的匹配线索，
        # V 是匹配后真正汇入当前位置的信息；三者都是从输入 x 学出来的。
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 所有头的结果拼回 C 维后，再通过一个可学习的线性变换。
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        # 下三角矩阵中，位置 t 只允许查看索引 <= t 的位置。
        # 例如前三个位置对应 [[1,0,0],[1,1,0],[1,1,1]]。
        # register_buffer 会让 mask 跟随模型搬到 CPU/CUDA，并进入 state_dict；
        # 但它不是需要训练的参数，因此优化器不会更新它。
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)))

    def forward(self, x):
        """x: [B,T,C] → 注意力输出: [B,T,C]。"""
        B, T, C = x.shape
        # self.qkv(x): [B,T,3*C]；沿最后一维切成三个 [B,T,C]。
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # 先把 C 拆成 H 个头、每头 D 维：[B,T,H,D]；
        # 再把 H 移到 T 前面：[B,H,T,D]，便于每个头独立计算注意力。
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # 每个 Query 与所有 Key 做点积，得到 [B,H,T,T] 的匹配分数。
        # 最后两个 T 分别表示「正在预测的位置」和「可参考的位置」。
        # 除以 sqrt(D) 可以避免维度变大时点积过大，让 softmax 更稳定。
        scores = q @ k.transpose(-2, -1) / self.head_size**0.5
        # mask[:T,:T] 只取当前序列所需的部分；未来位置被填成 -inf。
        # softmax 后这些位置的权重会变为 0，因此无法偷看后面的字符。
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        # softmax 把每一行分数变成权重，沿最后一维加起来为 1。
        # Dropout 只在训练模式生效；config.dropout=0 时不会丢弃权重。
        weights = self.dropout(F.softmax(scores, dim=-1))
        # 按权重对 Value 求加权和：[B,H,T,T] @ [B,H,T,D] → [B,H,T,D]。
        out = weights @ v
        # 把 H 个头移回字符维度旁，再拼成 [B,T,H*D] = [B,T,C]。
        # transpose 后内存可能不连续，contiguous() 让后面的 view 可用。
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """对每个字符位置分别做相同的非线性变换，不混合不同位置。"""

    def __init__(self, config):
        super().__init__()
        # nn.Sequential 表示依次执行这些层。最后一维先从 C 扩到 4C，
        # 再经 GELU 非线性激活，最后缩回 C，方便后续做残差相加。
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        """输入和输出的形状都是 [B,T,C]。"""
        return self.net(x)


class DecoderBlock(nn.Module):
    """一个 Decoder 块：先注意力，再 FFN；两部分都使用 Pre-Norm 和残差。"""

    def __init__(self, config):
        super().__init__()
        # LayerNorm 对每个位置的 C 个特征做归一化，形状仍为 [B,T,C]。
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x):
        """每次残差相加都保留原输入的信息，输出仍是 [B,T,C]。"""
        # Pre-Norm：先归一化，再注意力；然后把注意力结果加回原 x。
        # 注意力负责让不同位置交换信息。
        x = x + self.attn(self.ln1(x))
        # 再归一化、做 FFN，并把结果加回 x。
        # FFN 负责在每个位置内部变换特征。
        return x + self.ffn(self.ln2(x))


class GPT(nn.Module):
    """把 embedding、多个 DecoderBlock 和预测头连接成完整语言模型。"""

    def __init__(self, config, vocab_size):
        super().__init__()
        self.block_size = config.block_size
        # token 表：[V,C]。用整数 token ID 查表，得到字符向量。
        self.token_embedding = nn.Embedding(vocab_size, config.n_embd)
        # 位置表：[block_size,C]。位置 ID 0、1、2……也各有一个向量。
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        # 按 config.n_layer 堆叠 DecoderBlock；默认是 2 层。
        self.blocks = nn.Sequential(*(DecoderBlock(config) for _ in range(config.n_layer)))
        self.final_norm = nn.LayerNorm(config.n_embd)
        # 把每个位置的 C 维隐藏向量变成 V 个分数，分别对应 V 个字符。
        self.lm_head = nn.Linear(config.n_embd, vocab_size)

    def forward(self, input_ids, targets=None):
        """返回 logits [B,T,V]；若给了答案 targets，还返回标量 loss。"""
        # input_ids 是整数张量 [B,T]，不是原始字符串。
        B, T = input_ids.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.block_size}")
        # 位置 ID 为 0 到 T-1，必须与 input_ids 放在同一设备上。
        positions = torch.arange(T, device=input_ids.device)
        # token_embedding(input_ids): [B,T,C]；position_embedding(positions): [T,C]。
        # 相加时 PyTorch 会把位置向量广播到 B 个样本上，结果为 [B,T,C]。
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        # blocks 逐层处理；final_norm 不改变形状；lm_head 输出 [B,T,V]。
        # logits 是未经过 softmax 的原始分数，分数越大表示模型越倾向该字符。
        logits = self.lm_head(self.final_norm(self.blocks(x)))
        loss = None
        if targets is not None:
            # targets 为 [B,T]：每个输入位置的正确「下一个字符」ID。
            # cross_entropy 要求预测形状 [样本数,V]、答案形状 [样本数]，
            # 因此把 B*T 个预测位置展平；它内部会处理 softmax 和对数。
            # 默认得到 B*T 个位置的平均损失，一个标量。
            loss = F.cross_entropy(logits.reshape(B * T, -1), targets.reshape(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, greedy=False):
        """一次预测一个新字符，再把它接到输入末尾，重复指定次数。"""
        # @torch.no_grad() 表示生成时不建立反向传播所需的计算图，节省内存。
        if not greedy and temperature <= 0:
            raise ValueError("temperature must be positive")
        # eval() 关闭训练阶段的 Dropout；它不会删除或修改已学到的参数。
        self.eval()
        for _ in range(max_new_tokens):
            # 只把最近 block_size 个字符送进模型；更早的字符仍保留在输出里，
            # 但本次预测无法直接参考它们。输出 logits 的形状为 [B,T,V]。
            logits, _ = self(input_ids[:, -self.block_size:])
            # 只需要最后一个位置对「下一个字符」的 V 个分数：[B,V]。
            next_logits = logits[:, -1, :]
            if greedy:
                # 贪心：直接选分数最高的字符；同样输入通常得到同样结果。
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                # 温度越低，概率越集中；温度越高，选择越随机。
                # softmax 得到概率，multinomial 按概率抽取一个 ID。
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            # 把新 ID 接到末尾，下一轮便可以把刚生成的字符作为上下文。
            input_ids = torch.cat((input_ids, next_id), dim=1)
        # 返回包含原 prompt 和所有新字符的 ID 序列，形状 [B,原长度+新长度]。
        return input_ids
