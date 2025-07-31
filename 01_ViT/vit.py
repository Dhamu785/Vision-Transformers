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