import torch as t

import matplotlib.pyplot as plt
import json
import utils

with open('config.json', 'r') as f:
    config = json.load(f)
class PatchEmbeddings(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embd = t.nn.Conv2d(in_channels=config['num_channels'], 
                                        out_channels=config['embedding_dim'], stride=config['patch_size'],
                                        kernel_size=config['patch_size'])

    def forward(self, x:t.Tensor) -> t.Tensor:
        return t.flatten(self.patch_embd(x), 2).transpose(1,2)
    
class TransformerEncoder(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_norm1 = t.nn.LayerNorm(config['embedding_dim'])
        self.MHA = t.nn.MultiheadAttention(embed_dim=config['embedding_dim'], num_heads=config['attention_head'])
        self.layer_norm2 = t.nn.LayerNorm(config['embedding_dim'])
        self.mlp = t.nn.Sequential(
            t.nn.Linear(config['embedding_dim'], config['mlp_nodes']),
            t.nn.GELU(),
            t.nn.Linear(config['mlp_nodes'], config['embedding_dim'])
        )

    def forward(self, x:t.Tensor) -> t.Tensor:
        res1 = x
        x = self.MHA(self.layer_norm1(x)) + res1
        res2 = x
        x = self.mlp(self.layer_norm2(x)) + res2
        return x
