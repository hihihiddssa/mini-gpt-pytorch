from typing import cast

import torch
from torch import nn
from torch.nn import functional as F


# 因果自注意力机制
class CausalSelfAttention(nn.Module):
    # 调用类自动执行__init__方法
    def __init__(self, config):
        super().__init__()  # 访问父类的__init__方法 即执行nn.Module的__init__方法（也就是初始化）
        assert (
            config.n_embd % config.n_head == 0
        )  # 确保embedding维度可以被head数整除 assert是断言 如果不满足条件就会抛出异常
        self.n_head = config.n_head  # 定义多头注意力机制的头数
        self.head_size = (
            config.n_embd // config.n_head
        )  # 定义每个头的维度大小 也就是每个头看多少embedding维度 假设embedding维度是64 head数是4 那么每个头看16维度的embedding

        '''
        假设一个batch中的一句话 32个token 每个token64维度
        目前的矩阵是(32*64)的矩阵
        经过这个线性层之后得到的结果是(32*192)的矩阵
        这个(32*192)的矩阵呢 就是把每个token的64维度的embedding映射到了192维度的空间
        然后QKV三个矩阵一分为三 就是(32*64) (32*64) (32*64)
        从最开始的(32*64)到(32*192)再到(32*64) (32*64) (32*64)
        但是这个过程每个32*64的矩阵的每个元素都乘上了3个不同的权重 也就是QKV了 这个64*192的线性层就是权重矩阵了
        '''
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)

        # 多头注意力计算完成之后 需要对所有Head的结果进行一次线性变换 不会改变结构
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(config.block_size, config.block_size, dtype=torch.bool)
            ),
        )

    # 多头注意力机制的前向传播
    def forward(self, x):
        # 输入X
        # x.shape[16,32,64] B=16语料数量 T=32单句长度 C=64embedding维度
        B, T, C = x.shape


        #self.qkv(x)是一个线性层
        #[16,32,64] -> [16,32,192]
        #chunk(3, dim=-1):是将最后一维分成3份
        #[16,32,192] -> [16,32,64][16,32,64][16,32,64]
        #q k v 的结构分别都是[16,32,64]了
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # 照着head数量把q k v的结构分成多头
        # q=[16,32,64] ->[16,32,4,16] -> [16,4,32,16]最后那个16:每个head看16维度的embedding
        # n_head是多头头数 head_size是每个头看多少embedding维度 64维 4个头 每个头看16维度
        # 用上transpose是转换一下方便下一步计算
        # 到这里q k v的结构都是[16,4,32,16]了
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # 计算q和k的点积
        # q[16,4,32,16]
        # k[16,4,32,16] -> k.transpose(-2, -1) [16,4,16,32]
        # q @ k.transpose(-2, -1) [16,4,32,16] @ [16,4,16,32] -> scores[16,4,32,32]
        # 只看单个head的话 Q_head是[32,16] K_head是[32,16] Q_head @ K_head.T是[32,32] 也就是每个token和其他token的关系
        # 4个head的话就是[4,32,32] 也就是每个token和其他token的关系 4个不同的head看不同的关系
        # 再加上B=16就是[16,4,32,32] 也就是每个token和其他token的关系 4个不同的head看不同的关系 16个batch
        scores = q @ k.transpose(-2, -1) / self.head_size**0.5

        # 应用掩码 未来位置 score=-inf -> softmax之后=0
        mask = cast(torch.Tensor, self.mask)
        scores = scores.masked_fill(~mask[:T, :T], float("-inf"))

        # socres=[16,4,32,32] 对于某个batch的某个head的某个token 现在有一整排对其他token的分数
        # F.softmax(scores, dim=-1)会把scores归一化成概率分布 分数->概率/权重
        # self.dropout(F.softmax(scores, dim=-1))会随机丢弃一些权重 来做正则化regularization 防止模型训练过度依赖几个固定的注意力链接
        weights = self.dropout(F.softmax(scores, dim=-1))

        # [16,4,32,32] @ [16,4,32,16] -> [16,4,32,16]
        out = weights @ v
        # 把4个head的结果拼接起来 [16,4,32,16] -> [16,32,64]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # self.proj(out)是一个线性层
        # [16,32,64] @ [64,64] -> [16,32,64]
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # 非线性层
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

# 一个DecoderBlock的结构
class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)  # 创建n_embd维的LayerNorm层
        self.attn = CausalSelfAttention(config) # 创建CausalSelfAttention层
        self.ln2 = nn.LayerNorm(config.n_embd)  # 创建n_embd维的LayerNorm层
        self.ffn = FeedForward(config)          # 创建FeedForward层

    def forward(self, x):  # 进入的时候[B, T, C] B是batch_size T是block_size C是n_embd
        x = (
            x + self.attn(self.ln1(x))# 先执行layernorm 然后进入self.attn 经过CausalSelfAttention的forward 得到[B, T, C] 再残差链接加上原来的x 得到[B, T, C]
        )
        return x + self.ffn(self.ln2(x))# 同理


"\n前面有了几个零件：\n1. CausalSelfAttention: 实现了自注意力机制，带有掩码以确保因果性。\n2. FeedForward: 实现了前馈神经网络。\n3. DecoderBlock: 结合了LayerNorm、CausalSelfAttention和FeedForward，形成了一个完整的解码器块。\n\n现在我们将这些组件组合起来，构建一个完整的GPT模型。GPT模型包括：\n- token embedding层：将输入的token IDs映射到嵌入向量。\n- position embedding层：为每个位置添加位置信息。\n- 多个DecoderBlock：堆叠多个解码器块以增加模型的深度。\n- 最后的LayerNorm和线性层：将最终的嵌入向量映射到词汇表大小的logits，用于预测下一个token。\n"


class GPT(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.block_size = config.block_size
        self.token_embedding = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *(DecoderBlock(config) for _ in range(config.n_layer))
        )
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)

    def forward(self, input_ids, targets=None):
        B, T = ( input_ids.shape)  # B是batch_size一个batch包含的样本数，T是每个样本的长度，也是取样时的block_size

        # 检查输入的序列长度是否超过了模型的block_size，如果超过了就抛出异常
        if T > self.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block_size {self.block_size}"
            )

        # 创建当前序列的位置索引张量，范围是 [0, T)，用于位置嵌入
        positions = torch.arange(T, device=input_ids.device)

        # embedding+position_embedding
        # [B,T] -> [B,T,C] 也就是把每个token的ID映射到embedding向量上，并加上对应位置的位置信息
        x = self.token_embedding(input_ids) + self.position_embedding( positions)

        # x经过n个blocks->final_norm->lm_head 得到每个位置的logits
        # 假设509是vocab_size
        # 最后输出经过lm_head从[16,32,64] -> [16,32,509] 也就是每个位置的logits
        logits = self.lm_head(self.final_norm(self.blocks(x)))

        loss = None
        if targets is not None:
            # logits是[16,32,509] targets是正确答案shape是[16,32]
            # 每个位置的logits是一个长度为509的向量，表示该位置预测每个token的分数
            # CrossEntropy的输入是[样本数，类别数]因此看作总共有16*32=512个分类题 每一个题都有509个候选
            # 因此[B,T,V]->[B*T,V] [B,T]->[B*T] 其中V是vocab_size 而targetsreshape为[B*T]是每个位置的正确答案
            # 比如target=312 那么cross_entropy就会计算logits中第312个位置的分数高不高 分数高的话loss小
            loss = F.cross_entropy(logits.reshape(B * T, -1), targets.reshape(B * T))

        return (logits, loss)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, greedy=False):

        if not greedy and temperature <= 0:
            raise ValueError("temperature must be positive")
        self.eval()

        for _ in range(max_new_tokens):


            # self(...) 调用 GPT 的 forward()，将整个上下文输入模型，得到 logits 和 loss。
            # 例如输入 [青, 石, 镇]，shape=[1,3]，输出 logits.shape=[1,3,509]。
            # 由于使用 Causal Self-Attention，每个 Token 只能关注自己及之前的 Token：
            #   青的 logits：基于 [青]，预测下一个 Token
            #   石的 logits：基于 [青, 石]，预测下一个 Token
            #   镇的 logits：基于 [青, 石, 镇]，预测下一个 Token
            # 因此，虽然整个序列都输入了模型，但最后一个位置的 logits 已经融合了前面所有 Token 的信息。
            # 生成时只需要预测完整上下文后面的新 Token，所以只取最后一个位置的 logits。
            # 因此 next_logits = logits[:, -1, :]  # 取最后一个token的logits shape=[1,509]
            # 如果输入长度超过 block_size 假设为32，则只保留最后 32 个 Token 作为上下文。
            logits, _ = self(input_ids[:, -self.block_size :])
            next_logits = logits[:, -1, :]


            # 贪心策略 直接取预测概率最大值
            if greedy:
                next_id = next_logits.argmax(dim=-1, keepdim=True)

            # 采样策略 根据概率分布采样
            else:
                # dim=-1是对每个token的logits做softmax 得到概率分布
                # temperature是温度系数 控制采样的随机性 温度越高 softmax拉不开区别 温度越低 softmax拉开区别 也就是温度越低 越容易选到概率最大的token 即越保守
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)  # 根据概率分布采样一个token的ID shape=[1,1]

            # 把当前输出接入到输入中 继续预测下一个token shape=[1,4] [青,石,镇,下一个token]
            input_ids = torch.cat(
                (input_ids, next_id), dim=1
            )

        return input_ids
