# %%
from swin import swin_transformer
import torch as t
from config import Config
DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
# %%
x = t.randn(8, 3, 224, 224, device=DEVICE)
wa = swin_transformer(in_channels=Config.in_channels, hidden_dim=Config.hidden_dim, layers=Config.layers, 
                        downscaling_factor=Config.downscaling_factor, heads=Config.heads, 
                        head_dim=Config.head_dim, window_size=Config.window_size, 
                        relative_position=Config.relative_pos, num_clas=Config.relative_pos).to(DEVICE)
print(x.shape)
# %%
y = wa(x)
# %%
y.shape
# %%
