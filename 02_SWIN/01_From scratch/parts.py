import torch as t
from torch import nn

class PatchMerge(nn.Module):
    '''
        input shape = B, C, H, W
        output shape = B, H, W, C
    '''
    def __init__(self, in_channel: int, out_channels: int, downscaling_factor: int, bias:bool = False):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_features=in_channel*downscaling_factor**2, out_features=out_channels, bias=bias)

    def forward(self, x: t.Tensor) -> t.Tensor:
        b, c, h, w = x.shape
        new_h, new_w = h//self.downscaling_factor, w//self.downscaling_factor
        x = self.patch_merge(x).view(b, self.downscaling_factor, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)