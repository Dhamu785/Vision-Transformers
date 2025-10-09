import torch as t
from torch import nn
from typing import Tuple

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
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x

def get_mask(img_resolution: Tuple[int, int], window_size: int, shift: int, device: str):
    H,W = img_resolution
    mask = t.zeros(img_resolution, device=device, dtype=t.long)
    h_slice = (slice(0, -window_size), slice(-window_size, -shift), slice(-shift, None))
    w_slice = (slice(0, -window_size), slice(-window_size, -shift), slice(-shift, None))
    count = 0
    for h in h_slice:
        for w in w_slice:
            mask[h,w] = count
            count += 1
    mask_ids = mask.view(H // window_size, window_size, W // window_size, window_size).permute(0, 2, 1, 3).contiguous().view(-1, window_size * window_size)
    att_res = mask_ids.unsqueeze(1) - mask_ids.unsqueeze(2)
    att_mask = att_res.masked_fill(att_res == 0, float(0.0)).masked_fill(att_res != 0, float(-100.0))
    return att_mask

class WindowAttention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, shifted: int, window_size: int, rel_pos: bool):
        super().__init__()
        inner_dim = heads * head_dim
        self.scale = head_dim ** -0.5
        self.shifted = shifted
        self.window_size = window_size
        self.rel_pos = rel_pos
