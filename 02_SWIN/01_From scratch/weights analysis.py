# %% imports
import torch as t
import matplotlib.pyplot as plt
import numpy as np

from swin import swin_transformer
from config import Config

# %% load model
DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
mdl_pth = "C:\\Users\\dhamu\\Downloads\\SWIN\\Sample models\\numbers\\custom\\sample models\\mdl-10.pt"

model = swin_transformer(in_channels=Config.in_channels, hidden_dim=Config.hidden_dim, layers=Config.layers, 
                        downscaling_factor=Config.downscaling_factor, heads=Config.heads, 
                        head_dim=Config.head_dim, window_size=Config.window_size, 
                        relative_position=Config.relative_pos, num_clas=10).to(DEVICE)
model.load_state_dict(t.load(mdl_pth, map_location = DEVICE, weights_only = True), strict = False)
# %%
lyr1 = np.ravel(model.stage1.down_scale.linear.weight.to('cpu').detach().numpy())

# %%
lyr1
# %%
plt.hist(lyr1)
plt.show()
# %%
