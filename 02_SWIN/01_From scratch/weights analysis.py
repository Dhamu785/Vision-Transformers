# %% imports
import torch as t
import matplotlib.pyplot as plt
import os
from typing import Dict, Literal

from swin import swin_transformer
from config import Config

# %% defining functions
DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'

def get_weights(layer_name: str, num_classes: int, dataset_name: str):
    mdl_pth = f"C:\\Users\\dhamu\\Downloads\\SWIN\\Sample models\\{dataset_name}\\custom"
    mdls = os.listdir(mdl_pth)
    file_sorted = sorted(mdls, key=lambda x: int(x.split('-')[-1].split('.')[0]))
    weights = dict()
    for m in file_sorted:
        if 'pt' in m:
            mdl = os.path.join(mdl_pth, m)
            model = swin_transformer(in_channels=Config.in_channels, hidden_dim=Config.hidden_dim, layers=Config.layers, 
                            downscaling_factor=Config.downscaling_factor, heads=Config.heads, 
                            head_dim=Config.head_dim, window_size=Config.window_size, 
                            relative_position=Config.relative_pos, num_clas=num_classes).to(DEVICE)
            model.load_state_dict(t.load(mdl, map_location = DEVICE, weights_only = True), strict = False)
            state_dict = model.state_dict()
            weights[m.split('.')[0]] = state_dict[layer_name].detach().cpu().reshape(-1).tolist()
    return weights

def plot(type: Literal['hist', 'line_plot'], data: Dict, name: str, dataset_name: str):
    plt.figure(figsize=(15, 5))
    if type == 'hist':
        for i in data.keys():
            plt.hist(data[i], histtype='step', label='E-'+i.split('-')[1].split('.')[0], alpha=0.8)
            plt.legend(loc='upper right')
            plt.xlabel('Weights')
            plt.ylabel('Epochs')
            plt.title(f'{dataset_name} | Layer - {name}')
            plt.grid(visible=True, axis='both', which='both', color='#a1a1a1', linestyle='-', linewidth=1, mec='#00cefc')
        plt.show()
    elif type == 'line_plot':
        layer_no = 0
        plt.figure(figsize=(25,15))
        for i in range(1, 13):
            plt.subplot(3, 4, i)
            for j in range(layer_no, layer_no+8):
                plt.plot(range(0,110,10), data[layer_no], label=f'ly-{layer_no}')
                layer_no += 1
            plt.legend(loc='upper right')
            plt.suptitle(f'{dataset_name} | LayerNorm - {name}', y=0.93, x=0.51, fontsize=30)
        plt.show()

def multi_plot(type: Literal['hist', 'line_plot'], data: Dict, name: str, dataset_name: str, file_name: int, sav_loc=os.PathLike):
    plt.figure(figsize=(15, 5))
    if type == 'hist':
        for i in data.keys():
            plt.hist(data[i], histtype='step', label='E-'+i.split('-')[1].split('.')[0], alpha=0.8)
            plt.legend(loc='upper right')
            plt.xlabel('Weights')
            plt.ylabel('Epochs')
            plt.title(f'{dataset_name} | Layer - {name}')
            plt.grid(visible=True, axis='both', which='both', color='#a1a1a1', linestyle='-', linewidth=1, mec='#00cefc')
        sav = os.path.join(sav_loc, str(file_name)+'.png')
        plt.savefig(sav)
    elif type == 'line_plot':
        layer_no = 0
        plt.figure(figsize=(25,15))
        for i in range(1, 13):
            plt.subplot(3, 4, i)
            for j in range(layer_no, layer_no+8):
                plt.plot(range(0,110,10), data[layer_no], label=f'ly-{layer_no}')
                layer_no += 1
            plt.legend(loc='upper right')
            plt.suptitle(f'{dataset_name} | LayerNorm - {name}', y=0.93, x=0.51, fontsize=30)
        sav = os.path.join(sav_loc, str(file_name)+'.png')
        plt.savefig(sav)

def arrange_norm(data: Dict):
    layer_count = int(len(list(data.values())[0]))
    plotting_data = dict()
    for i in range(layer_count):
        plotting_data[i] = []
        for f in data.keys():
            plotting_data[i].append(data[f][i])

    return plotting_data

# %% Curve plot from histogram
# curve using bin centers
# for i in layer1.keys():
#     counts, bins = np.histogram(layer1[i], bins=10)
#     bin_centers = (bins[:-1] + bins[1:]) / 2
#     plt.plot(bin_centers, counts, label='E-'+i.split('-')[1].split('.')[0], alpha=0.5)
# plt.legend(loc='right')
# plt.show()

# %% stage1.down_scale.linear.weight
# dataset = 'food'
# num_classes = 211
dataset = 'numbers'
num_classes = 10
layer_name = 'stage1.down_scale.linear.weight'


weights = get_weights(layer_name = layer_name, num_classes=num_classes, dataset_name=dataset)
plot(type='hist', data=weights, name=layer_name, dataset_name=dataset)

# %% 'stage1.layers.0.0.layer_norm1.weight'
dataset = 'food'
num_classes = 211
# dataset = 'numbers'
# num_classes = 10
layer_name = 'stage1.layers.0.0.layer_norm1.weight'

weights = get_weights(layer_name=layer_name, num_classes=num_classes, dataset_name=dataset)
weights = arrange_norm(weights)
plot(type='line_plot', data=weights, name=layer_name, dataset_name=dataset)
# %% multiple plots
weights_layers = ['stage1.down_scale.linear.weight', 'stage1.layers.0.0.window_attn.qkv.weight',
            'stage1.layers.0.0.window_attn.to_out.weight', 'stage1.layers.0.0.window_attn.to_out.bias',
            'stage1.layers.0.0.mlp.0.weight', 'stage1.layers.0.0.mlp.0.bias', 'stage1.layers.0.0.mlp.2.weight',
            'stage1.layers.0.0.mlp.2.bias', 'stage1.layers.0.1.window_attn.qkv.weight', 
            'stage1.layers.0.1.window_attn.to_out.weight', 'stage1.layers.0.1.window_attn.to_out.bias',
            'stage1.layers.0.1.mlp.0.weight', 'stage1.layers.0.1.mlp.0.bias',
            'stage1.layers.0.1.mlp.2.weight', 'stage1.layers.0.1.mlp.2.bias']
norm_layers = ['stage1.layers.0.0.layer_norm1.weight', 'stage1.layers.0.0.layer_norm1.bias',
                'stage1.layers.0.0.layer_norm2.weight', 'stage1.layers.0.0.layer_norm2.bias',
                'stage1.layers.0.1.layer_norm1.weight', 'stage1.layers.0.1.layer_norm1.bias',
                'stage1.layers.0.1.layer_norm2.weight', 'stage1.layers.0.1.layer_norm2.bias']

stage2_wgt = ['stage2.down_scale.linear.weight', 'stage2.layers.0.0.window_attn.qkv.weight',
                'stage2.layers.0.0.window_attn.to_out.weight', 'stage2.layers.0.0.window_attn.to_out.bias',
                'stage2.layers.0.0.mlp.0.weight', 'stage2.layers.0.0.mlp.0.bias', 'stage2.layers.0.0.mlp.2.weight',
                'stage2.layers.0.0.mlp.2.bias', 'stage2.layers.0.1.window_attn.qkv.weight',
                'stage2.layers.0.1.window_attn.to_out.weight', 'stage2.layers.0.1.window_attn.to_out.bias',
                'stage2.layers.0.1.mlp.0.weight', 'stage2.layers.0.1.mlp.0.bias', 'stage2.layers.0.1.mlp.2.weight',
                'stage2.layers.0.1.mlp.2.bias']
stage2_norm = ['stage2.layers.0.0.layer_norm1.weight', 'stage2.layers.0.0.layer_norm1.bias',
                'stage2.layers.0.0.layer_norm2.weight', 'stage2.layers.0.0.layer_norm2.bias',
                'stage2.layers.0.1.layer_norm1.weight', 'stage2.layers.0.1.layer_norm1.bias',
                'stage2.layers.0.1.layer_norm2.weight', 'stage2.layers.0.1.layer_norm2.bias',]

stage3_wgt = ['stage3.down_scale.linear.weight', 'stage3.layers.0.0.window_attn.qkv.weight', 'stage3.layers.0.0.window_attn.to_out.weight',
                'stage3.layers.0.0.window_attn.to_out.bias', 'stage3.layers.0.0.mlp.0.weight', 'stage3.layers.0.0.mlp.0.bias',
                'stage3.layers.0.0.mlp.2.weight', 'stage3.layers.0.0.mlp.2.bias', 'stage3.layers.0.1.window_attn.qkv.weight',
                'stage3.layers.0.1.window_attn.to_out.weight', 'stage3.layers.0.1.window_attn.to_out.bias',
                'stage3.layers.0.1.mlp.0.weight', 'stage3.layers.0.1.mlp.0.bias', 'stage3.layers.0.1.mlp.2.weight',
                'stage3.layers.0.1.mlp.2.bias', 'stage3.layers.1.0.window_attn.qkv.weight', 'stage3.layers.1.0.window_attn.to_out.weight',
                'stage3.layers.1.0.window_attn.to_out.bias', 'stage3.layers.1.0.mlp.0.weight', 'stage3.layers.1.0.mlp.0.bias',
                'stage3.layers.1.0.mlp.2.weight', 'stage3.layers.1.0.mlp.2.bias', 'stage3.layers.1.1.window_attn.qkv.weight',
                'stage3.layers.1.1.window_attn.to_out.weight', 'stage3.layers.1.1.window_attn.to_out.bias',
                'stage3.layers.1.1.mlp.0.weight', 'stage3.layers.1.1.mlp.0.bias', 'stage3.layers.1.1.mlp.2.weight',
                'stage3.layers.1.1.mlp.2.bias', 'stage3.layers.2.0.window_attn.qkv.weight', 'stage3.layers.2.0.window_attn.to_out.weight',
                'stage3.layers.2.0.window_attn.to_out.bias', 'stage3.layers.2.0.mlp.0.weight', 'stage3.layers.2.0.mlp.0.bias',
                'stage3.layers.2.0.mlp.2.weight', 'stage3.layers.2.0.mlp.2.bias', 'stage3.layers.2.1.window_attn.qkv.weight',
                'stage3.layers.2.1.window_attn.to_out.weight', 'stage3.layers.2.1.window_attn.to_out.bias',
                'stage3.layers.2.1.mlp.0.weight', 'stage3.layers.2.1.mlp.0.bias', 'stage3.layers.2.1.mlp.2.weight',
                'stage3.layers.2.1.mlp.2.bias']
stage3_norm = ['stage3.layers.0.0.layer_norm1.weight', 'stage3.layers.0.0.layer_norm1.bias',
                'stage3.layers.0.0.layer_norm2.weight', 'stage3.layers.0.0.layer_norm2.bias',
                'stage3.layers.0.1.layer_norm1.weight', 'stage3.layers.0.1.layer_norm1.bias', 
                'stage3.layers.0.1.layer_norm2.weight', 'stage3.layers.0.1.layer_norm2.bias',
                'stage3.layers.1.0.layer_norm1.weight', 'stage3.layers.1.0.layer_norm1.bias',
                'stage3.layers.1.0.layer_norm2.weight', 'stage3.layers.1.0.layer_norm2.bias',
                'stage3.layers.1.1.layer_norm1.weight', 'stage3.layers.1.1.layer_norm1.bias',
                'stage3.layers.1.1.layer_norm2.weight', 'stage3.layers.1.1.layer_norm2.bias',
                'stage3.layers.2.0.layer_norm1.weight', 'stage3.layers.2.0.layer_norm1.bias',
                'stage3.layers.2.0.layer_norm2.weight', 'stage3.layers.2.0.layer_norm2.bias',
                'stage3.layers.2.1.layer_norm1.weight', 'stage3.layers.2.1.layer_norm1.bias',
                'stage3.layers.2.1.layer_norm2.weight', 'stage3.layers.2.1.layer_norm2.bias']

stage4_wgt = ['stage4.down_scale.linear.weight', 'stage4.layers.0.0.window_attn.qkv.weight',
                'stage4.layers.0.0.window_attn.to_out.weight', 'stage4.layers.0.0.window_attn.to_out.bias',
                'stage4.layers.0.0.mlp.0.weight', 'stage4.layers.0.0.mlp.0.bias', 'stage4.layers.0.0.mlp.2.weight',
                'stage4.layers.0.0.mlp.2.bias', 'stage4.layers.0.1.window_attn.qkv.weight',
                'stage4.layers.0.1.window_attn.to_out.weight', 'stage4.layers.0.1.window_attn.to_out.bias',
                'stage4.layers.0.1.mlp.0.weight', 'stage4.layers.0.1.mlp.0.bias', 'stage4.layers.0.1.mlp.2.weight',
                'stage4.layers.0.1.mlp.2.bias', 'mlp.2.weight', 'mlp.2.bias', 'mlp.3.weight', 'mlp.3.bias']
stage4_norm = ['stage4.layers.0.0.layer_norm1.weight', 'stage4.layers.0.0.layer_norm1.bias',
                'stage4.layers.0.0.layer_norm2.weight', 'stage4.layers.0.0.layer_norm2.bias',
                'stage4.layers.0.1.layer_norm1.weight', 'stage4.layers.0.1.layer_norm1.bias',
                'stage4.layers.0.1.layer_norm2.weight', 'stage4.layers.0.1.layer_norm2.bias',]

# %%  weights plot
dataset = 'numbers'
num_classes = 10
sav_loc = r'C:\Users\dhamu\Downloads\Plots\stage1-numbers_norm'

for i in range(len(stage4_wgt)):
    layer_name = stage4_wgt[i]

    weights = get_weights(layer_name = layer_name, num_classes=num_classes, dataset_name=dataset)
    multi_plot(type='hist', data=weights, name=layer_name, dataset_name=dataset, file_name=i, sav_loc=sav_loc)
# %% norm plot
dataset = 'numbers'
num_classes = 10
sav_loc = r'C:\Users\dhamu\Downloads\Plots\numbers\stage2-norm'

for i in range(len(stage2_norm)):
    layer_name = stage2_norm[i]

    weights = get_weights(layer_name = layer_name, num_classes=num_classes, dataset_name=dataset)
    arranged = arrange_norm(weights)
    multi_plot(type='line_plot', data=arranged, name=layer_name, dataset_name=dataset, file_name=i, sav_loc=sav_loc)
# %% Arranging the plots
import matplotlib.pyplot as plt
import cv2

plt.figure(figsize=(200, 60))
path = r'C:\Users\dhamu\Downloads\Plots\numbers\stage1-norm'
plots = os.listdir(path)
count = len(plots)
for i in range(count):
    img_data = cv2.imread(os.path.join(path, plots[i]))
    plt.subplot(2, 4, i+1)
    plt.axis('off')
    plt.imshow(img_data)

plt.show()