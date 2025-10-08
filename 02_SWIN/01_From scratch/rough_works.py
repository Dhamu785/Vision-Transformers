# %% Rollings imports==============================
import torch as t
# %% Rolling definition 
a = t.tensor([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]])
a = t.arange(1, 65).reshape(8,8)
print("Shape = ", a.shape)
print(a)
# %% Rolling implementation
rolled1 = t.roll(a, shifts=(2,2), dims=(0,1))
rolled2 = t.roll(a, shifts=(-2,-2), dims=(0,1))

print(rolled1)
print(rolled2)

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
x = t.tensor([[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]], dtype=t.float32)
downscaling_factor = 2
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
