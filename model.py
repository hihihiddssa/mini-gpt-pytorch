from typing import cast

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)#创建一个线性层
        self.proj = nn.Linear(config.n_embd, config.n_embd)#进入是[B, T, C] 输出是[B, T, C] 其中C是n_embd
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(config.block_size, config.block_size, dtype=torch.bool)
            ),
        )

    def forward(self, x):
        #[16,32,64]-> B=16 T=32 C=64
        B, T, C = x.shape
        #x.shape[16,32,64] B=16语料数量 T=32单句长度 C=64embedding维度
        
        #[16,32,192] -> [16,32,64][16,32,64][16,32,64]
        #self.qkv(x)是一个线性层:得到[64,192]的层 得到结果[16,32,192]
        #chunk(3, dim=-1):是将最后一维分成3份，得到q、k、v，每个都是[16,32,64]，其中64是n_embd
        #一个token有64维的embedding 经过（64,192）的线性层 得到192维度的向量 一组batch有16组句子 一组句子有32个token 所以得到[16,32,192]的张量 然后将最后一维分成3份 得到q、k、v 每个都是[16,32,64]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
       
        #q=[16,32,64] ->[16,32,4,16] -> [16,4,32,16] n_head是多头头数 head_size是每个头看多少embedding维度
        #用上transpose是方便下一步计算
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        #计算q和k的点积
        #q[16,4,32,16]
        #k[16,4,32,16] -> k.transpose(-2, -1) [16,4,16,32]
        #q @ k.transpose(-2, -1) [16,4,32,32] -> scores[16,4,32,32]
        #只看单个head的话 Q_head是[32,16] K_head是[32,16] Q_head @ K_head.T是[32,32] 也就是每个token和其他token的关系
        #4个head的话就是[4,32,32] 也就是每个token和其他token的关系 4个不同的head看不同的关系
        #再加上B=16就是[16,4,32,32] 也就是每个token和其他token的关系 4个不同的head看不同的关系 16个batch
        scores = q @ k.transpose(-2, -1) / self.head_size**0.5
        
        #应用掩码 未来位置 score=-inf -> softmax之后=0
        mask = cast(torch.Tensor, self.mask)
        scores = scores.masked_fill(
            ~mask[:T, :T],
            float("-inf")
        )

        #socres=[16,4,32,32] 对于某个batch的某个head的某个token 现在有一整排对其他token的分数
        #F.softmax(scores, dim=-1)会把scores归一化成概率分布 分数->概率/权重
        #self.dropout(F.softmax(scores, dim=-1))会随机丢弃一些权重 来做正则化regularization 防止模型训练过度依赖几个固定的注意力链接
        weights = self.dropout(F.softmax(scores, dim=-1))
        
        #[16,4,32,32] @ [16,4,32,16] -> [16,4,32,16]
        out = weights @ v
        #把4个head的结果拼接起来 [16,4,32,16] -> [16,32,64]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        #self.proj(out)是一个线性层:进入是[B, T, C] 输出是[B, T, C] 其中C是n_embd
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),#非线性
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)#创建n_embd维的LayerNorm层
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x):#进入的时候[B, T, C] B是batch_size T是block_size C是n_embd
        x = x + self.attn(self.ln1(x))#先执行layernorm 然后进入self.attn 经过CausalSelfAttention的forward 得到[B, T, C] 再加上原来的x 得到[B, T, C]
        return x + self.ffn(self.ln2(x))


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
        B, T = input_ids.shape#B是batch_size一个batch包含的样本数，T是每个样本的长度，也是取样时的block_size
        if T > self.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block_size {self.block_size}"
            )
        
        positions = torch.arange(T, device=input_ids.device)# 创建当前序列的位置索引张量，范围是 [0, T)，用于位置嵌入
        x = self.token_embedding(input_ids) + self.position_embedding(positions)#embedding

        #blocks的输出是[16,32,64]
        logits = self.lm_head(self.final_norm(self.blocks(x)))#经过blocks->final_norm->lm_head 得到每个位置的logits
        
        loss = None
        if targets is not None:
            #logits是[16,32,509] targets是[16,32] 其中509是vocab_size
            #一共有16个batch 每个batch有32个预测位置 每个位置都对509个token做预测 预测结果是一个概率分布
            #CrossEntropy的输入是[样本数，类别数]因此看作总共有512个分类题 每一个题都有509个候选
            #因此[B,T,V]->[B*T,V] [B,T]->[B*T] 其中V是vocab_size 而targetsreshape为[B*T]是每个位置的正确答案
            #比如target=312 那么cross_entropy就会计算logits中第312个位置的分数高不高 分数高的话loss小
            loss = F.cross_entropy(logits.reshape(B * T, -1), targets.reshape(B * T))#计算loss
        
        return (logits, loss)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, greedy=False):
        if not greedy and temperature <= 0:
            raise ValueError("temperature must be positive")
        self.eval()

        for _ in range(max_new_tokens):
            #关于-self.block_size: 如果输入的长度超过了模型的block_size，那么就只取最后block_size个token来预测下一个token，因为模型只能看到最近的block_size个token
            #假设input_ids是[青,石,镇]shape=[1,3] logits是[1,3,509] 但是只取最后最后的"镇"的logits来预测下一个token 因为前面两个token已经确定了
            logits, _ = self(input_ids[:, -self.block_size :])
            next_logits = logits[:, -1, :]#到这里得到[1,509] 也就是最后一个token的logits 也就是"镇"的logits

            if greedy:#贪心策略 直接取最大值
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:#采样策略 根据概率分布采样
                #dim=-1是对每个token的logits做softmax 得到概率分布
                #temperature是温度系数 控制采样的随机性 温度越高 softmax拉不开区别 温度越低 softmax拉开区别 也就是温度越低 越容易选到概率最大的token 即越保守
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)#根据概率分布采样一个token的ID shape=[1,1]
            input_ids = torch.cat((input_ids, next_id), dim=1)#把当前输出接入到输入中 继续预测下一个token shape=[1,4] [青,石,镇,下一个token]
        return input_ids
