# %% imports
import matplotlib.pyplot as plt
import pickle as pkl
from torchvision.utils import make_grid
# %%
pkl_pth = 'C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\03\\Vision-Transformers\\02_SWIN\\01_From scratch\\Hooks\\from_mdl-10.pkl'
with open(pkl_pth, 'rb') as f:
    data = pkl.load(f)

print(data.keys())
# %%
data['shifted']['output'].shape
# %%
