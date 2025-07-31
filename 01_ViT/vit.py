import torch as t

import matplotlib.pyplot as plt

import utils

class PatchEmbeddings(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embd = t.nn.Conv2d(in_channels=1)