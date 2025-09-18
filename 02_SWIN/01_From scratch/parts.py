import torch as t
from torch import nn

class PatchMerge(nn.Module):
    def __init__(self, in_channel: int, out_channels: int, downscaling_factor: int):
            super().__init__()
            self.patch_merge = nn.Unfold()