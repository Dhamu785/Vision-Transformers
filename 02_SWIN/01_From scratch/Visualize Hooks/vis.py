# %% imports
import matplotlib.pyplot as plt
import pickle as pkl
from torchvision.utils import make_grid, save_image
import os
import shutil
# %%
pkl_pth = 'C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\03\\Vision-Transformers\\02_SWIN\\01_From scratch\\Hooks\\from_mdl-100.pkl'
with open(pkl_pth, 'rb') as f:
    data = pkl.load(f)

print(data.keys())
# %%
print(f"Shaped for\nInput = {data['input'].shape}\nResized = {data['resized'].shape}")
print(f"Shifted = {data['shifted']['output'].shape}\nNo shift = {data['no shift']['output'].shape}")

# %% View input image

## To plot the grid
# grid = make_grid(data['input'], 2, 3, pad_value=2, normalize=True).moveaxis(0, 2)
# plt.subplots(figsize=(5,10))
# plt.axis('off')
# plt.imshow(grid, cmap='gray')

grid = make_grid(data['input'], 2, 3, pad_value=2, normalize=True)
save_image(grid, 'test_samples.png')
# %% resized view
sav_loc = os.path.join(os.getcwd(), 'resized')
if os.path.exists(sav_loc):
    shutil.rmtree(sav_loc)
os.mkdir(sav_loc)

for i in range(48):
    grid = make_grid(data['resized'].permute(0,3,1,2)[:,i,:,:].unsqueeze(1), nrow=2, padding=3, pad_value=2, normalize=True)
    save_image(grid, os.path.join(sav_loc, f'resized-f{i}.png'))
# %% Saving the no shifts
hooks_pth = 'C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\03\\Vision-Transformers\\02_SWIN\\01_From scratch\\Hooks'
hooks_lst = os.listdir(hooks_pth)
sav_loc_no_shift = os.path.join(os.getcwd(), 'no shift')

if os.path.exists(sav_loc_no_shift):
    shutil.rmtree(sav_loc_no_shift)
os.mkdir(sav_loc_no_shift)
Filters = 96
for i in range(Filters):
    filter_save = os.path.join(sav_loc_no_shift, f"Filter-{i}")
    os.mkdir(filter_save)

for i in range(len(hooks_lst)):
    epoch = hooks_lst[i].split('-')[1].split('.')[0]
    with open(os.path.join(hooks_pth, hooks_lst[i]), 'rb') as f:
        data = pkl.load(f)

    for j in range(Filters):
        grid = make_grid(data['no shift']['output'].permute(0,3,1,2)[:,j,:,:].unsqueeze(1), nrow=2, padding=3, pad_value=2, normalize=True)
        save_image(grid, os.path.join(os.path.join(sav_loc_no_shift, f"Filter-{j}"), f'Epoch-{i}.png'))
# %% No shift
# %% Saving the no shifts
hooks_pth = 'C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\03\\Vision-Transformers\\02_SWIN\\01_From scratch\\Hooks'
hooks_lst = os.listdir(hooks_pth)
sav_loc_no_shift = os.path.join(os.getcwd(), 'shifted')

if os.path.exists(sav_loc_no_shift):
    shutil.rmtree(sav_loc_no_shift)
os.mkdir(sav_loc_no_shift)
Filters = 96
for i in range(Filters):
    filter_save = os.path.join(sav_loc_no_shift, f"Filter-{i}")
    os.mkdir(filter_save)

for i in range(len(hooks_lst)):
    epoch = hooks_lst[i].split('-')[1].split('.')[0]
    with open(os.path.join(hooks_pth, hooks_lst[i]), 'rb') as f:
        data = pkl.load(f)

    for j in range(Filters):
        grid = make_grid(data['shifted']['output'].permute(0,3,1,2)[:,j,:,:].unsqueeze(1), nrow=2, padding=3, pad_value=2, normalize=True)
        save_image(grid, os.path.join(os.path.join(sav_loc_no_shift, f"Filter-{j}"), f'Epoch-{i}.png'))
# %%
grid.shape
# %%
f, axs = plt.subplots(figsize=(5,10))
axs.set_axis_off()
axs.imshow(grid, cmap='binary')

# %%
grid.min(), grid.max()

# %%
