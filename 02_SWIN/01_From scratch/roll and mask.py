# %%
import torch as t
# %%
window_size = 7
shift = window_size // 2
img_h, img_w = 14, 14
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
