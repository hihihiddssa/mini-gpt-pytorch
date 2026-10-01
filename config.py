from dataclasses import asdict, dataclass

@dataclass
class Config:
    block_size: int = 32
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.0
    batch_size: int = 16
    max_steps: int = 50000
    eval_interval: int = 250
    eval_batches: int = 20
    learning_rate: float = 0.0003
    seed: int = 42

    def to_dict(self):
        return asdict(self)
