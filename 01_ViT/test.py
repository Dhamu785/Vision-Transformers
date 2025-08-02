# %%
from vit import ViT
import torch as t
# %%
x = t.randn(8, 1, 28, 28)
print(x.shape)
# %%
model = ViT()
# %%
y = model(x)
# %%
y.shape
# %%
