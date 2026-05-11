from dataclasses import dataclass


@dataclass
class Config:
    vocab_size: int = 0
    data_dir: str = "./data/shakespeare_char"
    n_embd: int = 768
    batch_size: int = 12
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = False
    device: str = "cuda"
    n_layer: int = 12
    n_head: int = 12

    max_iters: int = 10000
    eval_interval: int = 500
    eval_iters: int = 100
    learning_rate: float = 3e-4
