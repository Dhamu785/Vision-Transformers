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

# %% stage1.layers.0.0.window_attn.qkv.weight
# dataset = 'food'
# num_classes = 211
dataset = 'numbers'
num_classes = 10
layer_name = 'stage1.layers.0.0.window_attn.qkv.weight'


weights = get_weights(layer_name=layer_name, num_classes=num_classes, dataset_name=dataset)
plot(type='hist', data=weights, name=layer_name, dataset_name=dataset)

# %% stage1.layers.0.0.window_attn.to_out.weight
# dataset = 'food'
# num_classes = 211
dataset = 'numbers'
num_classes = 10
layer_name = 'stage1.layers.0.0.window_attn.to_out.weight'

weights = get_weights(layer_name=layer_name, num_classes=num_classes, dataset_name=dataset)
plot(type='hist', data=weights, name=layer_name, dataset_name=dataset)

# %% stage1.layers.0.0.window_attn.to_out.bias
dataset = 'food'
num_classes = 211
# dataset = 'numbers'
# num_classes = 10
layer_name = 'stage1.layers.0.0.window_attn.to_out.bias'

weights = get_weights(layer_name=layer_name, num_classes=num_classes, dataset_name=dataset)
plot(type='hist', data=weights, name=layer_name, dataset_name=dataset)

# %% stage1.layers.0.0.mlp.0.weight
# dataset = 'food'
# num_classes = 211
dataset = 'numbers'
num_classes = 10
layer_name = 'stage1.layers.0.0.mlp.0.weight'

weights = get_weights(layer_name=layer_name, num_classes=num_classes, dataset_name=dataset)
plot(type='hist', data=weights, name=layer_name, dataset_name=dataset)
# %% stage1.layers.0.0.mlp.0.bias
dataset = 'food'
num_classes = 211
# dataset = 'numbers'
# num_classes = 10
layer_name = 'stage1.layers.0.0.mlp.0.bias'

weights = get_weights(layer_name=layer_name, num_classes=num_classes, dataset_name=dataset)
plot(type='hist', data=weights, name=layer_name, dataset_name=dataset)

# %% stage1.layers.0.0.mlp.2.weight
# dataset = 'food'
# num_classes = 211
dataset = 'numbers'
num_classes = 10
layer_name = 'stage1.layers.0.0.mlp.2.weight'

weights = get_weights(layer_name=layer_name, num_classes=num_classes, dataset_name=dataset)
plot(type='hist', data=weights, name=layer_name, dataset_name=dataset)

# %% stage1.layers.0.0.mlp.2.bias
dataset = 'food'
num_classes = 211
# dataset = 'numbers'
# num_classes = 10
layer_name = 'stage1.layers.0.0.mlp.2.bias'

weights = get_weights(layer_name=layer_name, num_classes=num_classes, dataset_name=dataset)
plot(type='hist', data=weights, name=layer_name, dataset_name=dataset)
# %% stage1.layers.0.0.layer_norm2.weight
# dataset = 'food'
# num_classes = 211
dataset = 'numbers'
num_classes = 10
layer_name = 'stage1.layers.0.0.layer_norm2.weight'

weights = get_weights(layer_name=layer_name, num_classes=num_classes, dataset_name=dataset)
weights = arrange_norm(weights)
plot(type='line_plot', data=weights, name=layer_name, dataset_name=dataset)

# %% stage1.layers.0.0.layer_norm2.bias
dataset = 'food'
num_classes = 211
# dataset = 'numbers'
# num_classes = 10
layer_name = 'stage1.layers.0.0.layer_norm2.bias'

weights = get_weights(layer_name=layer_name, num_classes=num_classes, dataset_name=dataset)
weights = arrange_norm(weights)
plot(type='line_plot', data=weights, name=layer_name, dataset_name=dataset)

# %% stage2.down_scale.linear.weight
# dataset = 'food'
# num_classes = 211
dataset = 'numbers'
num_classes = 10
layer_name = 'stage2.down_scale.linear.weight'

weights = get_weights(layer_name=layer_name, num_classes=num_classes, dataset_name=dataset)
plot(type='hist', data=weights, name=layer_name, dataset_name=dataset)

# %% stage3.down_scale.linear.weight
dataset = 'food'
num_classes = 211
# dataset = 'numbers'
# num_classes = 10
layer_name = 'stage3.down_scale.linear.weight'

weights = get_weights(layer_name=layer_name, num_classes=num_classes, dataset_name=dataset)
plot(type='hist', data=weights, name=layer_name, dataset_name=dataset)

# %% stage4.down_scale.linear.weight
# dataset = 'food'
# num_classes = 211
dataset = 'numbers'
num_classes = 10
layer_name = 'stage4.down_scale.linear.weight'

weights = get_weights(layer_name=layer_name, num_classes=num_classes, dataset_name=dataset)
plot(type='hist', data=weights, name=layer_name, dataset_name=dataset)

# %% mlp.2.weight
# dataset = 'food'
# num_classes = 211
dataset = 'numbers'
num_classes = 10
layer_name = 'mlp.2.weight'

weights = get_weights(layer_name=layer_name, num_classes=num_classes, dataset_name=dataset)
plot(type='hist', data=weights, name=layer_name, dataset_name=dataset)

# %% mlp.2.bias
# dataset = 'food'
# num_classes = 211
dataset = 'numbers'
num_classes = 10
layer_name = 'mlp.3.bias'

weights = get_weights(layer_name=layer_name, num_classes=num_classes, dataset_name=dataset)
plot(type='hist', data=weights, name=layer_name, dataset_name=dataset)

# %%
