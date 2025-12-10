# %% config path
import sys
import os

sys.path.append(os.path.join(os.getcwd(), '02_SWIN', '01_From scratch'))
# %% imports
import os
import numpy as np
import pickle as pkl
from PIL import Image

from config import Config
from swin import swin_transformer

import torch as t
from torchvision import transforms

DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def collect_img(img_path):
    tensor_lst = []
    imgs = os.listdir(img_path)
    for i in imgs:
        tensor_lst.append(transform(Image.open(os.path.join(img_path, i))))
    return t.from_numpy(np.array(tensor_lst))

sample_path = 'C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\03\\Vision-Transformers\\02_SWIN\\01_From scratch\\sample_dataset'
test = collect_img(sample_path)

# %% Functions to record filters
def input1(mdl, inp, out):
    layers['input1'] = inp[0].detach().cpu()

def input2(mdl, inp, out):
    layers['input2'] = out[0].detach().cpu()

def no_shift(mdl, inp, out):
    layers['no shift']['output'] = out.detach().cpu()

def shifted(mdl, inp, out):
    layers['shifted']['output'] = out.detach().cpu()

swin_model = swin_transformer(in_channels=Config.in_channels, hidden_dim=Config.hidden_dim, layers=Config.layers, 
                        downscaling_factor=Config.downscaling_factor, heads=Config.heads, 
                        head_dim=Config.head_dim, window_size=Config.window_size, 
                        relative_position=Config.relative_pos, num_clas=Config.num_class).to(DEVICE)

# %%
def get_filters(model_path):
    swin_model.load_state_dict(t.load(os.path.join(model_path), map_location=DEVICE, weights_only=True), strict=False)

    swin_model.stage1.down_scale.patch_merge.register_forward_hook(input1)
    swin_model.stage1.down_scale.linear.register_forward_hook(input2)
    swin_model.stage1.layers[0][0].window_attn.to_out.register_forward_hook(no_shift)

    with t.inference_mode():
        swin_model(test.to(DEVICE))
# %%
model_path = "C:\\Users\\dhamu\\Downloads\\SWIN\\mdl-7.pt"
models = os.listdir(model_path)
for i in models:
    layers = {'no shift': {}, 'shifted': {}}
    mp = os.path.join(model_path, i)
    epoch = i.split('-')[1].split('.')[0]
    if int(epoch) % 10 == 0:
        get_filters(model_path=model_path)
        with open(f'Hooks\\from_mdl-{epoch}.pkl', 'wb') as f:
            pkl.dump(layers, f)
# %%
layers['input1'].shape
# %%
layers['input2'].shape

# %%
