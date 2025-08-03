# %%
from vit import ViT
import torch as t
import utils
# %%
model = ViT()
train, val = utils.get_loaders(32)
# %%
batch = next(iter(train))
utils.plot_data(batch)
# %%
optimizer = t.optim.Adam(model.parameters(), lr=0.01)
loss = t.nn.CrossEntropyLoss()
# %%
model_pack = utils.model_utils('cpu', optimizer, loss, train, val, 20)
# %%
info = model_pack.train_model(model)
# %%
