# %%
import torch as t
# %% Rolling definition 
a = t.arange(0, 64).reshape(1,8,8)
print("Shape = ", a.shape)
print(a)
# %% Rolling implementation
shift = 2
rolled1 = t.roll(a, shifts=(shift,shift), dims=(1,2))
rolled2 = t.roll(a, shifts=(-shift,-shift), dims=(1,2))

print(rolled1)
print(rolled2)
# %% My implementation testings =============================
from typing import Tuple
import torch as t
from einops import rearrange
# %%
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
    mask_ids = rearrange(mask, "(nh wh) (nw ww) -> (nh nw) (wh ww)", wh=window_size, ww=window_size).contiguous()
    att_res = mask_ids.unsqueeze(1) - mask_ids.unsqueeze(2)
    att_mask = att_res.masked_fill(att_res == 0, float(0.0)).masked_fill(att_res != 0, float(-100.0))
    return att_mask # no. of windows, window size ** 2, window size ** 2
# %%
test = get_mask((8,8), 4, 2, 'cpu')
print(test.shape)
# %%
for i in range(4):
    print(test[i])
# %%
