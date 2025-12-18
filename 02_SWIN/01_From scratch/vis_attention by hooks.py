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

# %% Functions to record filters (Hooks factory)
def get_hook(name: str, capture_input: bool) -> t.Tensor:
    def hook(model, inpt, outpt):
        data = inpt[0] if capture_input else outpt
        layers[name] = data.detach().cpu()

# %%
def get_filters(model_path):
    swin_model = swin_transformer(in_channels=Config.in_channels, hidden_dim=Config.hidden_dim, layers=Config.layers, 
                        downscaling_factor=Config.downscaling_factor, heads=Config.heads, 
                        head_dim=Config.head_dim, window_size=Config.window_size, 
                        relative_position=Config.relative_pos, num_clas=Config.num_class).to(DEVICE)
    
    swin_model.load_state_dict(t.load(model_path, map_location=DEVICE, weights_only=True), strict=False)

    with t.inference_mode():
        swin_model(test.to(DEVICE))
# %%
model_path = "C:\\Users\\dhamu\\Downloads\\SWIN\\Model samples"
models = os.listdir(model_path)
for i in models:
    layers = {'no shift': {}, 'shifted': {}}
    mp = os.path.join(model_path, i)
    epoch = i.split('-')[1].split('.')[0]
    if int(epoch) % 10 == 0:
        get_filters(model_path=mp)
        with open(f'Hooks\\from_mdl-{epoch}.pkl', 'wb') as f:
            pkl.dump(layers, f)
# %%
layers['input'].shape
# %%
layers['resized'].shape

# %%
