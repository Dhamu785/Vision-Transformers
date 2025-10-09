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
# x = t.randn(8,3,224,224)
# x = t.tensor([[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]], dtype=t.float32)
x = t.arange(0, 64*3, dtype=t.float16).reshape(1,3,8,8)
print(x.shape)
print(x[0])
# %%
downscaling_factor = 4
print(x.shape)
# %%
unfold = t.nn.Unfold(kernel_size=downscaling_factor, padding=0, stride=downscaling_factor)
un = unfold(x[0])
print(un.shape)
print(un[20])
# print(un)
# %%
y = un.view(8, -1, 56, 56).permute(0, 2, 3, 1)
print(y)