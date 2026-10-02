from dataclasses import asdict, dataclass

@dataclass
class Config:
    block_size: int = 128
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 3
    dropout: float = 0.1
    batch_size: int = 16
    max_steps: int = 200000
    eval_interval: int = 250
    eval_batches: int = 20
    learning_rate: float = 0.0003
    seed: int = 42

    def to_dict(self):
        return asdict(self)
