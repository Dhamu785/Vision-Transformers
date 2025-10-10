import torch as t
from torch import nn
from typing import Tuple
from einops import rearrange, einsum
import numpy as np

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
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
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
    return att_mask # no. of windows, window size ** 2

class WindowAttention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, shifted: int, shift:int, window_size: int, rel_pos: bool, device: str):
        super().__init__()
        inner_dim = heads * head_dim
        self.scale = head_dim ** -0.5
        self.shifted = shifted
        self.window_size = window_size
        self.rel_pos = rel_pos
        self.shift = shift
        self.heads = heads
        if shifted:
            mask = get_mask(img_resolution=(512, 512), window_size=window_size, shift=shift, device=device)
            self.register_buffer("attention mask", mask)
        if rel_pos:
            indices = t.from_numpy(np.array([x, y] for x in range(self.window_size) for y in range(self.window_size)))
            self.abs_distance = (indices[None, :, :] - indices[:, None, :]) + (window_size-1)
            self.ref_tab = nn.Parameter(t.randn(2 * window_size -1, t.randn(2 * window_size - 1)))
        else:
            self.ref_tab = nn.Parameter(t.randn(window_size**2, window_size**2))
            
        self.qkv = nn.Linear(in_features = dim, out_features = inner_dim*3, bias=False)

    def forward(self, x: t.Tensor) -> t.Tensor:
        if self.shifted:
            x = t.roll(x, shifts=(-self.shift, -self.shift), dims=(1,2))

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t,'B (h1 wh) (w1 ww) (h d) -> (B h1 w1) h (wh ww) d',
                                            h=self.heads, wh=self.window_size, ww=self.window_size), qkv)
        qk = einsum(q, k, 'b h w1 d, b h w2 d -> b h w1 w2') * self.scale

        if self.rel_pos:
            qk += self.ref_tab[self.abs_distance[:,:,0], self.abs_distance[:,:,1]]
        else:
            qk += self.ref_tab

        if self.shifted:
            ...