import torch as t

import matplotlib.pyplot as plt
import json
import utils
import configurations

info = configurations.config()
class PatchEmbeddings(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embd = t.nn.Conv2d(in_channels=info.num_channels, 
                                        out_channels=info.embedding_dim, stride=info.patch_size,
                                        kernel_size=info.patch_size)

    def forward(self, x:t.Tensor) -> t.Tensor:
        return t.flatten(self.patch_embd(x), 2).transpose(1,2)
    
class TransformerEncoder(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_norm1 = t.nn.LayerNorm(info.embedding_dim)
        self.MHA = t.nn.MultiheadAttention(embed_dim=info.embedding_dim, num_heads=info.attention_head)
        self.layer_norm2 = t.nn.LayerNorm(info.embedding_dim)
        self.mlp = t.nn.Sequential(
            t.nn.Linear(info.embedding_dim, info.mlp_nodes),
            t.nn.GELU(),
            t.nn.Linear(info.mlp_nodes, info.embedding_dim)
        )

    def forward(self, x:t.Tensor) -> t.Tensor:
        res1 = x
        x = self.MHA(self.layer_norm1(x)) + res1
        res2 = x
        x = self.mlp(self.layer_norm2(x)) + res2
        return x

class MLP(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = t.nn.LayerNorm(info.embedding_dim)
        self.linear_layers = t.nn.Sequential(
            t.nn.Linear(info.embedding_dim, info.mlp_nodes),
            t.nn.ReLU(),
            t.nn.Linear(info.mlp_nodes, info.num_classes)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.linear_layers(self.norm(x))
        return x
