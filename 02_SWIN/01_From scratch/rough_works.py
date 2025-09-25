# %% Rollings imports==============================
import torch as t
# %% Rolling definition 
a = t.tensor([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]])
print("Shape = ", a.shape)
print(a)
# %% Rolling implementation
rolled1 = t.roll(a, shifts=(2,2), dims=(0,1))
rolled2 = t.roll(rolled1, shifts=(-2,-2), dims=(0,1))

print(rolled1)
print(rolled2)
# %% Mask analysis==================================
import torch as t
from einops import rearrange
# %%
window_size = 3
mask = t.zeros((window_size**2, window_size**2))
mask.shape, mask
# %%
displacement = 1
mask[-displacement * window_size:, :-displacement * window_size] = float('-inf') # mask[-3:, :-3]
print(mask)
mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')
print(mask)

# %% Rearrange mask=======
window_size = 3
displacement = 1
mask = t.zeros((window_size**2, window_size**2))
mask.shape, mask

# %%
mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
mask[:, -displacement:, :, :-displacement] = float('-inf')
mask[:, :-displacement, :, -displacement:] = float('-inf')
mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')
print(mask)
print(mask.shape)
# %%
import torch

H, W = 6, 6      # feature map size
window_size = 3   # window size

# Create a grid of window IDs
window_ids = torch.zeros((H, W), dtype=torch.int)
win_id = 0
for i in range(0, H, window_size):
    for j in range(0, W, window_size):
        window_ids[i:i+window_size, j:j+window_size] = win_id
        win_id += 1

print(window_ids)

# %%
shift = window_size // 2  # usually half window
shifted_ids = torch.roll(window_ids, shifts=(-shift, -shift), dims=(0,1))
print(shifted_ids)

# %%
def get_exact_mask(shifted_window_ids):
    """
    shifted_window_ids: torch tensor of shape (window_size, window_size)
    returns: mask of shape (window_size**2, window_size**2)
    """
    ws2 = window_size ** 2
    flat_ids = shifted_window_ids[0:3, 0:3].flatten()  # shape: ws2
    print(flat_ids)
    mask = torch.zeros((ws2, ws2))
    
    # Compare IDs
    for i in range(ws2):
        for j in range(ws2):
            if flat_ids[i] != flat_ids[j]:
                mask[i, j] = float('-inf')
    return mask

# %%
# Example usage in attention
mask = get_exact_mask(shifted_ids)
# In scaled dot-product attention:
# attn = softmax((Q @ K.T)/sqrt(d) + mask) @ V
print(mask)
# %% Relative distance==================================
import torch as t
import numpy as np
# %%
def get_relative_distances(window_size):
    indices = t.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    # print(indices)
    distances = indices[None, :, :] - indices[:, None, :]
    return distances
# %%
distance = get_relative_distances(7)+6
print(distance)
print(distance.shape)
# %%
print(distance.min(), distance.max())
# %% Unfold ==================================
import torch as t
# %%
x = t.randn(8,3,224,224)
# x = t.tensor([[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]], dtype=t.float32)
downscaling_factor = 4
print(x.shape)
# %%
unfold = t.nn.Unfold(kernel_size=downscaling_factor, padding=0, stride=downscaling_factor)
un = unfold(x)
print(un.shape)
# print(un)
# %%
y = un.view(8, -1, 56, 56).permute(0, 2, 3, 1)
print(y)
# %%
y.shape
# %%
2*7-1
# %%
