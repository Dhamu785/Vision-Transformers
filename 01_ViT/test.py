# %%
from vit import ViT
import torch as t
import utils
# %%
model = ViT()
train, val = utils.get_loaders(32)
device = 'cuda' if t.cuda.is_available() else 'cpu'
# %%
batch = next(iter(train))
utils.plot_data(batch)
# %%
optimizer = t.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
loss = t.nn.CrossEntropyLoss()
# %%
model_pack = utils.model_utils(device, optimizer, loss, train, val, 2, False)
# %%
info = model_pack.train_model(model)
# %%
