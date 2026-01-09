# %% imports
import torch as t
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from swin import swin_transformer
from config import Config

# %% load model
DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
mdl_pth = "C:\\Users\\dhamu\\Downloads\\SWIN\\Sample models\\food\\custom"
mdls = os.listdir(mdl_pth)

layer1 = dict()
for m in mdls:
    mdl = os.path.join(mdl_pth, m)
    model = swin_transformer(in_channels=Config.in_channels, hidden_dim=Config.hidden_dim, layers=Config.layers, 
                        downscaling_factor=Config.downscaling_factor, heads=Config.heads, 
                        head_dim=Config.head_dim, window_size=Config.window_size, 
                        relative_position=Config.relative_pos, num_clas=211).to(DEVICE)
    model.load_state_dict(t.load(mdl, map_location = DEVICE, weights_only = True), strict = False)
    layer1[m] = np.ravel(model.stage1.down_scale.linear.weight.to('cpu').detach().numpy())
# %%
plt.figure(figsize=(15,5))
for i in layer1.keys():
    plt.hist(layer1[i], histtype='step', label='E-'+i.split('-')[1].split('.')[0], alpha=0.8)
plt.legend()
plt.xlabel('Weights')
plt.ylabel('Epochs')
plt.title('Food | Layer-1')
plt.grid(visible=True, axis='both', which='both', color='#a1a1a1', linestyle='-', linewidth=1, mec='#00cefc')
plt.show()

# %% curve
for i in layer1.keys():
    counts, bins = np.histogram(layer1[i], bins=10)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(bin_centers, counts, label='E-'+i.split('-')[1].split('.')[0], alpha=0.5)
plt.legend(loc='right')
plt.show()