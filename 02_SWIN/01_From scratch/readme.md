# **Input / Output of every layers**
## **SWIN - tiny outline**
### **01. Parameters**
1. `Parameters for SwinTransformer class`
    - hidden_dim = 96
    - layers = (2, 2, 6, 2)
    - heads = (3, 6, 12, 24)
    - chanels = 3
    - num_classes = 1000
    - head_dim = 32
    - window_size = 7
    - downscaling_factor = (4, 2, 2, 2)
    - relative_pos_embedding = True
    - `StageModule class params`
        - in_channels = 3 (channels)
        - hidden_dimensions = 96
        - layers = 2 (layers[0])
        - downscaling_factor = 4 (fownscaling_factor[0])
        - num_heads = 3 (heads[0])
        - head_dim = 32
        - window_size = 7
        - relative_pos_embeddings = True
        - `Patch_Merging params`
            - in_channels = 3
            - out_channels = 96 (hidden_dimensions)
            - downscaling_factor = 4
        - `SwinBlock params`
            - dim = 96
            - heads = 3
            - head_dim = 32
            - mlp_dim = 96(hidde_dimensions) * 4
            - shifted = True / False
            - window_size = 7
            - relative_pos_embedding = True
            - `Window_attention params`
                - dim = 96
                - heads = 3
                - head_dim = 32
                - shifted = True / False
                - window_size = 7
                - relative_pos_embedding = True

### **02. Input / Output shapes**
#### 01. Patch_Merging
- `params :` in_channel=3, out_channels=96, downscaling_factor=4
- `Input shape :` (B, C, H, W)
    - `unfold shape =` (B, C*(downscaling_factor**2), H/4 * W/4) -> (Batch, channels, total_patches)
- `Output shape :` (B, H/4, W/4, C)
#### 02. Window attention
- `params :` dim=96, heads=3, head_dim=32, shifted=True/False, window_size=7, relative_pos_embedding=True
- Input shape :` (B, H/4, w/4, C) => (8, H/4, W/4, 96)