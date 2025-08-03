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
import utils
# %%
trian, test = utils.get_loaders(32)
# %%
len(trian)
# %%
next(iter(trian))[1]
# %%
