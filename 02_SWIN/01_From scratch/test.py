# %%
from parts import WindowAttention
import torch as t
# %%
x = t.randn(8, 56, 56, 3)
wa = WindowAttention(dim=3, heads=3, head_dim=4, shifted=True, shift=2, window_size=7, rel_pos=True)
print(x.shape)
# %%
y = wa(x)
# %%
y.shape
# %%
