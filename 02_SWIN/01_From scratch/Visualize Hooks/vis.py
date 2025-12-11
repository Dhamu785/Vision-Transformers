# %% imports
import matplotlib.pyplot as plt
import pickle as pkl
from torchvision.utils import make_grid, save_image
# %%
pkl_pth = 'C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\03\\Vision-Transformers\\02_SWIN\\01_From scratch\\Hooks\\from_mdl-10.pkl'
with open(pkl_pth, 'rb') as f:
    data = pkl.load(f)

print(data.keys())
# %%
print(f"Shaped for\ninput = {data['input'].shape}\nno resized = {data['resized'].shape}")
print(f"Shaped for shifted = {data['shifted']['output'].shape}\nno shift = {data['no shift']['output'].shape}")

# %% View input image

## To plot the grid
# grid = make_grid(data['input'], 2, 3, pad_value=2, normalize=True).moveaxis(0, 2)
# plt.subplots(figsize=(5,10))
# plt.axis('off')
# plt.imshow(grid, cmap='gray')

grid = make_grid(data['input'], 2, 3, pad_value=2, normalize=True)
save_image(grid, 'test_samples.png')
# %% resized view
grid = make_grid(data['resized'].permute(0,3,1,2)[:,2,:,:].unsqueeze(1), 2, 3, pad_value=2, normalize=True).moveaxis(0, 2)

# %% Shifted
grid = make_grid(data['shifted']['output'].permute(0,3,1,2)[:,0,:,:].unsqueeze(1), 2, 3, pad_value=2, normalize=True).moveaxis(0, 2)

# %% No shift
grid = make_grid(data['no shift']['output'].permute(0,3,1,2)[:,0,:,:].unsqueeze(1), 2, 3, pad_value=2, normalize=True).moveaxis(0, 2)
# %%
grid.shape
# %%
f, axs = plt.subplots(figsize=(5,10))
axs.set_axis_off()
axs.imshow(grid, cmap='binary')

# %%
grid.min(), grid.max()

# %%
