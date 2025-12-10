# %% config path
import sys
import os

sys.path.append(os.path.join(os.getcwd(), '02_SWIN', '01_From scratch'))
# %% import modules
from swin import swin_transformer
import torch as t
from config import Config

DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
# %%
x = t.randn(8, 3, 224, 224, device=DEVICE)
model = swin_transformer(in_channels=Config.in_channels, hidden_dim=Config.hidden_dim, layers=Config.layers, 
                        downscaling_factor=Config.downscaling_factor, heads=Config.heads, 
                        head_dim=Config.head_dim, window_size=Config.window_size, 
                        relative_position=Config.relative_pos, num_clas=Config.num_class).to(DEVICE)
# %%
model_path = "C:\\Users\\dhamu\\Downloads\\SWIN\\mdl-7.pt"
model.load_state_dict(t.load(os.path.join(model_path), map_location=DEVICE, weights_only=True), strict=False)
# %%
model
# %%
model.stage1.layers[0][0].window_attn.to_out
# %%
