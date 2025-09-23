# %%
import torch as t
# %%
window_size = 4
shift = window_size // 2
img_h, img_w = 8, 8
# %%
img_mask = t.zeros((img_h, img_w))
print(img_mask.shape)
# %%
h_slices = (slice(0, -window_size),
            slice(-window_size, -shift),
            slice(-shift, None))
w_slices = (slice(0, -window_size),
            slice(-window_size, -shift),
            slice(-shift, None))
# %%
count = 0
for h in h_slices:
    for w in w_slices:
        img_mask[h, w] = count
        count += 1

print(img_mask)
# %%
print(img_mask.shape)
# %%
mask_windows = img_mask.view(img_h // window_size, window_size, img_w // window_size, window_size)
print(mask_windows.shape)
print(mask_windows)
# %%
mask_windows = mask_windows.permute(0, 2, 1, 3).contiguous().view(-1,window_size * window_size)
print(mask_windows)
# %%
attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
print(attn_mask)
# %%
attn_mask[1]
# %%
a = t.tensor([[[1,2,3,4]],[[5,6,7,8]]])
b = t.tensor([[1,2,3,4],[5,6,7,8]])
c = a-b.unsqueeze(2)
print(c.shape)
print(c)