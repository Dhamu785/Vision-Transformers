# %%
from swin import swin_transformer
import torch as t
# %%
x = t.randn(8, 3, 224, 224)
wa = swin_transformer(in_channels=3, hidden_dim=96, layers=[2,2,6,2], downscaling_factor=[4,2,2,2], heads=(3,6,12,24), 
                        head_dim=32, window_size=7, relative_position=True, num_clas=10)
print(x.shape)
# %%
y = wa(x)
# %%
y.shape
# %%
