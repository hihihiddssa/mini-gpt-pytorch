from dataclasses import asdict, dataclass


@dataclass
class Config:
    block_size: int = 32
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.0
    batch_size: int = 16
    max_steps: int = 300
    eval_interval: int = 50
    eval_batches: int = 10
    learning_rate: float = 3e-4
    seed: int = 42

    def to_dict(self):
        return asdict(self)
