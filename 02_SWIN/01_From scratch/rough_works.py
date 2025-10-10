# %% Relative distance==================================
import torch as t
import numpy as np
# %%
window_size = 7
indices = t.from_numpy(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
distance = (indices[None, :, :] - indices[:, None, :]) + (window_size-1)
# print(abs_distance)
print("Minimum value = ", distance.min().item(), "Maximum value = ", distance.max().item(), "Shape = ", distance.shape)
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