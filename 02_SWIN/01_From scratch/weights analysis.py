# %% imports
import torch as t
import matplotlib.pyplot as plt
import numpy as np
import os

from swin import swin_transformer
from config import Config

# %% Giving the path
DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
mdl_pth = "C:\\Users\\dhamu\\Downloads\\SWIN\\Sample models\\numbers\\custom"
# mdl_pth = "/Users/dhamodharan/My-Python/AI-Tutorials/SWIN supportings/Sample models/food/custom"
mdls = os.listdir(mdl_pth)

# %% Load model and plotting layer1
layer1 = dict()
for m in mdls:
    mdl = os.path.join(mdl_pth, m)
    if 'pt' in mdl:
        model = swin_transformer(in_channels=Config.in_channels, hidden_dim=Config.hidden_dim, layers=Config.layers, 
                            downscaling_factor=Config.downscaling_factor, heads=Config.heads, 
                            head_dim=Config.head_dim, window_size=Config.window_size, 
                            relative_position=Config.relative_pos, num_clas=10).to(DEVICE)
        model.load_state_dict(t.load(mdl, map_location = DEVICE, weights_only = True), strict = False)
        layer1[m] = np.ravel(model.stage1.down_scale.linear.weight.to('cpu').detach().numpy())

plt.figure(figsize=(15,5))
for i in layer1.keys():
    plt.hist(layer1[i], histtype='step', label='E-'+i.split('-')[1].split('.')[0], alpha=0.8)
plt.legend(loc='upper right')
plt.xlabel('Weights')
plt.ylabel('Epochs')
plt.title('SVHN | Layer-1')
plt.grid(visible=True, axis='both', which='both', color='#a1a1a1', linestyle='-', linewidth=1, mec='#00cefc')
plt.show()

# %% curve using bin centers
for i in layer1.keys():
    counts, bins = np.histogram(layer1[i], bins=10)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(bin_centers, counts, label='E-'+i.split('-')[1].split('.')[0], alpha=0.5)
plt.legend(loc='right')
plt.show()

# %% Normalization analysis
# mdl_path = "C:\\Users\\dhamu\\Downloads\\SWIN\\Sample models\\numbers\\custom\\mdl-100.pt"
mdl_path = "/Users/dhamodharan/My-Python/AI-Tutorials/SWIN supportings/Sample models/numbers/custom/mdl-10.pt"
model = swin_transformer(in_channels=Config.in_channels, hidden_dim=Config.hidden_dim, layers=Config.layers, 
                        downscaling_factor=Config.downscaling_factor, heads=Config.heads, 
                        head_dim=Config.head_dim, window_size=Config.window_size, 
                        relative_position=Config.relative_pos, num_clas=10).to(DEVICE)
model.load_state_dict(t.load(mdl_path, map_location = DEVICE, weights_only = True), strict = False)
# %%
weights = model.stage1.down_scale.linear.weight

# %%
print(weights.tolist())
# %%
