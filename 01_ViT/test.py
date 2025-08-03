# %%
from vit import ViT
import torch as t
import utils
# %%
model = ViT()
# %%
train, val = utils.get_loaders(32)
# %%
optimizer = t.optim.Adam(model.parameters(), lr=0.001)
loss = t.nn.CrossEntropyLoss()
# %%
model_pack = utils.model_utils('cpu', optimizer, loss, train, val, 2)
# %%
info = model_pack.train_model(model)
# %%
