from typing import Tuple
class Config:
    batch_size = 8
    in_channels: int = 3
    hidden_dim: int = 96
    layers: Tuple[int, ...] = (2,2,6,2)
    downscaling_factor: Tuple[int, ...] = (4,2,2,2)
    heads: Tuple[int, ...] = (3,6,12,24)
    head_dim: int = 32
    window_size: int = 7
    relative_pos: bool = True
    num_class: int = 10