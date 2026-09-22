"""把模型结构和训练参数放在同一个地方，方便阅读和调整。"""

from dataclasses import asdict, dataclass


@dataclass
class Config:
    """一组参数；@dataclass 会自动生成接收这些参数的 __init__ 方法。"""

    # 一次最多让模型看多少个连续字符。训练时 x/y 的第二维 T 就是它。
    # 生成时若已有文字超过 32 个字符，只把最后 32 个字符送入模型。
    block_size: int = 32

    # 每个字符在模型内部用多少个数字表示，常记作 C 或 embedding 维度。
    n_embd: int = 64
    # 注意力分成多少个头。本项目每头维度是 n_embd // n_head = 16。
    n_head: int = 4
    # 连续堆叠多少个 DecoderBlock。
    n_layer: int = 2
    # Dropout 比例；0.0 表示当前没有随机丢弃神经元输出。
    dropout: float = 0.0

    # 每次参数更新使用多少段训练文本，常记作 batch 维度 B。
    batch_size: int = 16
    # 不传 --steps 时默认更新参数的次数；训练更多步可用命令行覆盖。
    max_steps: int = 300
    # 每隔多少步打印一次平均 loss。
    eval_interval: int = 50
    # 每次打印平均 loss 时，随机抽取多少个 batch 求平均。
    # 当前这些 batch 仍来自训练文本，并非独立验证集。
    eval_batches: int = 10
    # AdamW 每一步调整参数的基本幅度；3e-4 等于 0.0003。
    learning_rate: float = 3e-4
    # 固定随机种子，让初始化和抽样在相同环境中较容易复现。
    seed: int = 42

    def to_dict(self):
        """保存 checkpoint 时，把 Config 转成普通字典。"""
        return asdict(self)
