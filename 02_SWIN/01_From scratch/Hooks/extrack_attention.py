# %% imports
import os
import numpy as np
import pickle as pkl
from PIL import Image

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

# %%
