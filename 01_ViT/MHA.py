import torch as t
import torch.nn as nn

class mha(nn.Module):
    def __init__(self, dim: int, num_heads: int, 
                    qkv_bias: bool, att_drop: float, 
                    proj_drop: float) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
