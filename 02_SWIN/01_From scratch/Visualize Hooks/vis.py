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
data['input2'].shape
# %%
# grid = make_grid(data['input1'][:,2,:,:].unsqueeze(1), 2, 3, pad_value=2, normalize=True).moveaxis(0, 2)
grid = make_grid(data['input1'][:,:,:], 2, 3, pad_value=2, normalize=True).moveaxis(0, 2)

# %%
grid.shape
# %%
f, axs = plt.subplots(figsize=(5,10))
axs.set_axis_off()
# plt.close()
axs.imshow(grid, cmap='gray')
# %%
grid.min(), grid.max()

# %%
