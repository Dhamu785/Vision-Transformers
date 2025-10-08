# %% Rollings imports==============================
import torch as t
# %% Rolling definition 
# a = t.tensor([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]])
a = t.arange(1, 65).reshape(8,8)
print("Shape = ", a.shape)
print(a)
# %% Rolling implementation
rolled1 = t.roll(a, shifts=(2,2), dims=(0,1))
rolled2 = t.roll(a, shifts=(-2,-2), dims=(0,1))

print(rolled1)
print(rolled2)
# %% Mask for shifted window ====================================
def window_partition(x, window_size: int):
    """
    Partitions the input tensor into non-overlapping windows.

    Args:
        x (torch.Tensor): Input tensor of shape (B, H, W, C).
        window_size (int): The size of the attention window.

    Returns:
        torch.Tensor: A tensor of shape (B * num_windows, window_size, window_size, C)
                        where windows are stacked in the batch dimension.
    """
    B, H, W, C = x.shape
    # Reshape the tensor to group pixels into windows: (B, H/ws, ws, W/ws, ws, C)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # Permute to bring window dimensions together: (B, H/ws, W/ws, ws, ws, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    # Reshape to stack all windows in the batch dimension
    windows = windows.view(-1, window_size, window_size, C)
    return windows

def create_swin_attention_mask(input_resolution, window_size, shift_size, device):
    """
    Creates the Swin Transformer attention mask for shifted windows.
    This mask prevents attention between non-adjacent local windows.

    Args:
        input_resolution (tuple[int]): The height and width of the input feature map (H, W).
        window_size (int): The size of the attention window.
        shift_size (int): The size of the cyclic shift. If 0, no mask is returned.
        device (torch.device): The device to create the mask on.

    Returns:
        torch.Tensor or None: The attention mask of shape (num_windows, window_area, window_area)
                                Returns None if shift_size is 0.
    """
    # If shift_size is 0, no masking is needed.
    if shift_size == 0:
        return None

    H, W = input_resolution
    
    # 1. Create a canvas (image mask) with unique IDs for different regions.
    img_mask = t.zeros((1, H, W, 1), device=device, dtype=t.long)
    
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    # 2. Partition the image mask into windows.
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)

    # 3. Create the attention mask from the partitioned image mask.
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

    # 4. Fill the mask with a large negative value where regions are different.
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    
    return attn_mask
# %%
mask = create_swin_attention_mask((8,8), 4, 2, 'cpu')
# print(mask[0])
print(mask[0])
# %% Relative distance==================================
import torch as t
import numpy as np
# %%
def get_relative_distances(window_size):
    indices = t.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    # print(indices)
    distances = indices[None, :, :] - indices[:, None, :]
    return distances
# %%
distance = get_relative_distances(7)+6
print(distance)
print(distance.shape)
# %%
print(distance.min(), distance.max())
# %% Unfold ==================================
import torch as t
# %%
# x = t.randn(8,3,224,224)
# x = t.tensor([[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]], dtype=t.float32)
x = t.arange(0, 64*3, dtype=t.float16).reshape(1,3,8,8)
print(x.shape)
print(x[0])
# %%
downscaling_factor = 4
print(x.shape)
# %%
unfold = t.nn.Unfold(kernel_size=downscaling_factor, padding=0, stride=downscaling_factor)
un = unfold(x[0])
print(un.shape)
print(un[20])
# print(un)
# %%
y = un.view(8, -1, 56, 56).permute(0, 2, 3, 1)
print(y)
# %%
y.shape
# %%
2*7-1
# %%
