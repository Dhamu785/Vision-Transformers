# %% import libaries
from swin import swin_transformer
from config import Config
import os
import torch as t
from torchvision import transforms
from torch.profiler import tensorboard_trace_handler, profile, ProfilerActivity, schedule
from PIL import Image

# %% Call model and load state dict
DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
model = swin_transformer(in_channels=Config.in_channels, hidden_dim=Config.hidden_dim, layers=Config.layers, 
                        downscaling_factor=Config.downscaling_factor, heads=Config.heads, 
                        head_dim=Config.head_dim, window_size=Config.window_size, 
                        relative_position=Config.relative_pos, num_clas=10).to(DEVICE)

model_path = "C:\\Users\\dhamu\\Downloads\\SWIN\\Sample models\\numbers\\custom\\mdl-10.pt"
model.load_state_dict(t.load(os.path.join(model_path), map_location=DEVICE, weights_only=True), strict=False)

# %% Image handling
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img_pth = "C:\\Users\\dhamu\\Downloads\\SWIN\\sample_dataset\\numbers\\7.jpg"
img = Image.open(img_pth)
img_t = transform(img).unsqueeze(0).repeat(64,1,1,1).to(DEVICE)

# %% Profiler
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=0, warmup=1, active=3),
                on_trace_ready=tensorboard_trace_handler('.\\logs\\batch64\\profile'),
                record_shapes=True, profile_memory=True, with_flops=True,
                with_stack=True, with_modules=True) as prof:
    with t.inference_mode():
        for i in range(10):
            model(img_t)
            prof.step()
# %%
print(prof.key_averages().table(
    sort_by="cpu_time_total",
    row_limit=200
))
# %%
