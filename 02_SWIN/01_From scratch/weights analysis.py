# %% imports
import torch as t
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict, Literal

from swin import swin_transformer
from config import Config

# %% Giving the path
DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'

# %% defining functions
mdl_pth = "C:\\Users\\dhamu\\Downloads\\SWIN\\Sample models\\numbers\\custom"
mdls = os.listdir(mdl_pth)

def get_weights(layer_name: str):
    weights = dict()
    for m in mdls:
        if 'pt' in m:
            mdl = os.path.join(mdl_pth, m)
            model = swin_transformer(in_channels=Config.in_channels, hidden_dim=Config.hidden_dim, layers=Config.layers, 
                            downscaling_factor=Config.downscaling_factor, heads=Config.heads, 
                            head_dim=Config.head_dim, window_size=Config.window_size, 
                            relative_position=Config.relative_pos, num_clas=10).to(DEVICE)
            model.load_state_dict(t.load(mdl, map_location = DEVICE, weights_only = True), strict = False)
            state_dict = model.state_dict()
            weights[layer_name] = state_dict[layer_name].detach().cpu().tolist()
    return weights

def plot(type: Literal['hist', 'line_plot'], data: Dict, name: str):
    plt.figure(figsize=(15, 5))
    if type == 'hist':
        for i in data.keys():
            plt.hist(data[i], histtype='step', label='E-'+i.split('-')[1].split('.')[0], alpha=0.8)
            plt.legend(loc='upper right')
            plt.xlabel('Weights')
            plt.ylabel('Epochs')
            plt.title(f'SVHN | Layer-1 | {name}')
            plt.grid(visible=True, axis='both', which='both', color='#a1a1a1', linestyle='-', linewidth=1, mec='#00cefc')
            plt.show()

# %% curve using bin centers
for i in layer1.keys():
    counts, bins = np.histogram(layer1[i], bins=10)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(bin_centers, counts, label='E-'+i.split('-')[1].split('.')[0], alpha=0.5)
plt.legend(loc='right')
plt.show()

# %% Normalization analysis - Loading the model
layer_norm1 = dict()
mdl_path = "C:\\Users\\dhamu\\Downloads\\SWIN\\Sample models\\numbers\\custom"
# mdl_path = "/Users/dhamodharan/My-Python/AI-Tutorials/SWIN supportings/Sample models/numbers/custom"
models = os.listdir(mdl_path)
file_sorted = sorted(models, key=lambda x: int(x.split('-')[-1].split('.')[0]))
for m in file_sorted:
    if 'pt' in m:
        mdl = os.path.join(mdl_path, m)
        model = swin_transformer(in_channels=Config.in_channels, hidden_dim=Config.hidden_dim, layers=Config.layers, 
                                downscaling_factor=Config.downscaling_factor, heads=Config.heads, 
                                head_dim=Config.head_dim, window_size=Config.window_size, 
                                relative_position=Config.relative_pos, num_clas=10).to(DEVICE)
        model.load_state_dict(t.load(mdl, map_location = DEVICE, weights_only = True), strict = False)

        gamma = model.stage1.layers[0][0].layer_norm1.weight.detach().cpu().tolist()
        beta = model.stage1.layers[0][0].layer_norm1.bias.detach().cpu().tolist()
        layer_norm1[m] = {'gamma': gamma, 'beta':beta}

# %% Arranging the data
plotting_data_gamma = dict()
plotting_data_beta = dict()

layers = 96
for l in range(layers):
    plotting_data_gamma[l] = []
    plotting_data_beta[l] = []
    for f in file_sorted:
        plotting_data_gamma[l].append(layer_norm1[f]['gamma'][l])
        plotting_data_beta[l].append(layer_norm1[f]['beta'][l])

# layer_norm1['mdl-10.pt']['gamma'][0]
# %% Plotting the data
layer_no = 0
plt.figure(figsize=(25,15))
for i in range(1, 13):
    plt.subplot(3, 4, i)
    for j in range(layer_no, layer_no+8):
        plt.plot(range(0,110,10), plotting_data_gamma[layer_no], label=f'ly-{layer_no}')
        layer_no += 1
    plt.legend(loc='upper right')
    plt.suptitle('SVHN | LayerNorm1 (gamma)', y=0.93, x=0.51, fontsize=30)
plt.show()
# %%
path = "C:\\Users\\dhamu\\Downloads\\SWIN\\Sample models\\numbers\\custom\\mdl-100.pt"
model = swin_transformer(in_channels=Config.in_channels, hidden_dim=Config.hidden_dim, layers=Config.layers, 
                                downscaling_factor=Config.downscaling_factor, heads=Config.heads, 
                                head_dim=Config.head_dim, window_size=Config.window_size, 
                                relative_position=Config.relative_pos, num_clas=10).to(DEVICE)
model.load_state_dict(t.load(path, map_location = DEVICE, weights_only = True), strict = False)
weights = model.stage1.layers[0][0].window_attn.qkv.weight.detach().cpu().tolist()

# %%
list(model.state_dict().keys())
# %%
