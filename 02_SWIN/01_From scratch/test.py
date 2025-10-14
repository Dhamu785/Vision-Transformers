# %%
from parts import Stages
import torch as t
# %%
x = t.randn(8, 3, 224, 224)
wa = Stages(in_dim=3, hidden_dim=96, layers=2, downscaling_factor=4, heads=3, head_dim=4, window_size=7, rel_pos=True)
print(x.shape)
# %%
y = wa(x)
# %%
y.shape
# %%
