# **Info about the model**
## **01. SWIN - tiny outline**
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
- `Input shape :` (B, H/4, w/4, C) => (8, H/4, W/4, 96)
- `Output shape :` (8, H/4, W/4, 96)

# **02. State dict**
['stage1.down_scale.linear.weight',
 'stage1.layers.0.0.layer_norm1.weight',
 'stage1.layers.0.0.layer_norm1.bias',
 'stage1.layers.0.0.window_attn.ref_tab',
 'stage1.layers.0.0.window_attn.rel_pos_1D',
 'stage1.layers.0.0.window_attn.qkv.weight',
 'stage1.layers.0.0.window_attn.to_out.weight',
 'stage1.layers.0.0.window_attn.to_out.bias',
 'stage1.layers.0.0.mlp.0.weight',
 'stage1.layers.0.0.mlp.0.bias',
 'stage1.layers.0.0.mlp.2.weight',
 'stage1.layers.0.0.mlp.2.bias',
 'stage1.layers.0.0.layer_norm2.weight',
 'stage1.layers.0.0.layer_norm2.bias',
 'stage1.layers.0.1.layer_norm1.weight',
 'stage1.layers.0.1.layer_norm1.bias',
 'stage1.layers.0.1.window_attn.ref_tab',
 'stage1.layers.0.1.window_attn.rel_pos_1D',
 'stage1.layers.0.1.window_attn.qkv.weight',
 'stage1.layers.0.1.window_attn.to_out.weight',
 'stage1.layers.0.1.window_attn.to_out.bias',
 'stage1.layers.0.1.mlp.0.weight',
 'stage1.layers.0.1.mlp.0.bias',
 'stage1.layers.0.1.mlp.2.weight',
 'stage1.layers.0.1.mlp.2.bias',
 'stage1.layers.0.1.layer_norm2.weight',
 'stage1.layers.0.1.layer_norm2.bias',
 'stage2.down_scale.linear.weight',
 'stage2.layers.0.0.layer_norm1.weight',
 'stage2.layers.0.0.layer_norm1.bias',
 'stage2.layers.0.0.window_attn.ref_tab',
 'stage2.layers.0.0.window_attn.rel_pos_1D',
 'stage2.layers.0.0.window_attn.qkv.weight',
 'stage2.layers.0.0.window_attn.to_out.weight',
 'stage2.layers.0.0.window_attn.to_out.bias',
 'stage2.layers.0.0.mlp.0.weight',
 'stage2.layers.0.0.mlp.0.bias',
 'stage2.layers.0.0.mlp.2.weight',
 'stage2.layers.0.0.mlp.2.bias',
 'stage2.layers.0.0.layer_norm2.weight',
 'stage2.layers.0.0.layer_norm2.bias',
 'stage2.layers.0.1.layer_norm1.weight',
 'stage2.layers.0.1.layer_norm1.bias',
 'stage2.layers.0.1.window_attn.ref_tab',
 'stage2.layers.0.1.window_attn.rel_pos_1D',
 'stage2.layers.0.1.window_attn.qkv.weight',
 'stage2.layers.0.1.window_attn.to_out.weight',
 'stage2.layers.0.1.window_attn.to_out.bias',
 'stage2.layers.0.1.mlp.0.weight',
 'stage2.layers.0.1.mlp.0.bias',
 'stage2.layers.0.1.mlp.2.weight',
 'stage2.layers.0.1.mlp.2.bias',
 'stage2.layers.0.1.layer_norm2.weight',
 'stage2.layers.0.1.layer_norm2.bias',
 'stage3.down_scale.linear.weight',
 'stage3.layers.0.0.layer_norm1.weight',
 'stage3.layers.0.0.layer_norm1.bias',
 'stage3.layers.0.0.window_attn.ref_tab',
 'stage3.layers.0.0.window_attn.rel_pos_1D',
 'stage3.layers.0.0.window_attn.qkv.weight',
 'stage3.layers.0.0.window_attn.to_out.weight',
 'stage3.layers.0.0.window_attn.to_out.bias',
 'stage3.layers.0.0.mlp.0.weight',
 'stage3.layers.0.0.mlp.0.bias',
 'stage3.layers.0.0.mlp.2.weight',
 'stage3.layers.0.0.mlp.2.bias',
 'stage3.layers.0.0.layer_norm2.weight',
 'stage3.layers.0.0.layer_norm2.bias',
 'stage3.layers.0.1.layer_norm1.weight',
 'stage3.layers.0.1.layer_norm1.bias',
 'stage3.layers.0.1.window_attn.ref_tab',
 'stage3.layers.0.1.window_attn.rel_pos_1D',
 'stage3.layers.0.1.window_attn.qkv.weight',
 'stage3.layers.0.1.window_attn.to_out.weight',
 'stage3.layers.0.1.window_attn.to_out.bias',
 'stage3.layers.0.1.mlp.0.weight',
 'stage3.layers.0.1.mlp.0.bias',
 'stage3.layers.0.1.mlp.2.weight',
 'stage3.layers.0.1.mlp.2.bias',
 'stage3.layers.0.1.layer_norm2.weight',
 'stage3.layers.0.1.layer_norm2.bias',
 'stage3.layers.1.0.layer_norm1.weight',
 'stage3.layers.1.0.layer_norm1.bias',
 'stage3.layers.1.0.window_attn.ref_tab',
 'stage3.layers.1.0.window_attn.rel_pos_1D',
 'stage3.layers.1.0.window_attn.qkv.weight',
 'stage3.layers.1.0.window_attn.to_out.weight',
 'stage3.layers.1.0.window_attn.to_out.bias',
 'stage3.layers.1.0.mlp.0.weight',
 'stage3.layers.1.0.mlp.0.bias',
 'stage3.layers.1.0.mlp.2.weight',
 'stage3.layers.1.0.mlp.2.bias',
 'stage3.layers.1.0.layer_norm2.weight',
 'stage3.layers.1.0.layer_norm2.bias',
 'stage3.layers.1.1.layer_norm1.weight',
 'stage3.layers.1.1.layer_norm1.bias',
 'stage3.layers.1.1.window_attn.ref_tab',
 'stage3.layers.1.1.window_attn.rel_pos_1D',
 'stage3.layers.1.1.window_attn.qkv.weight',
 'stage3.layers.1.1.window_attn.to_out.weight',
 'stage3.layers.1.1.window_attn.to_out.bias',
 'stage3.layers.1.1.mlp.0.weight',
 'stage3.layers.1.1.mlp.0.bias',
 'stage3.layers.1.1.mlp.2.weight',
 'stage3.layers.1.1.mlp.2.bias',
 'stage3.layers.1.1.layer_norm2.weight',
 'stage3.layers.1.1.layer_norm2.bias',
 'stage3.layers.2.0.layer_norm1.weight',
 'stage3.layers.2.0.layer_norm1.bias',
 'stage3.layers.2.0.window_attn.ref_tab',
 'stage3.layers.2.0.window_attn.rel_pos_1D',
 'stage3.layers.2.0.window_attn.qkv.weight',
 'stage3.layers.2.0.window_attn.to_out.weight',
 'stage3.layers.2.0.window_attn.to_out.bias',
 'stage3.layers.2.0.mlp.0.weight',
 'stage3.layers.2.0.mlp.0.bias',
 'stage3.layers.2.0.mlp.2.weight',
 'stage3.layers.2.0.mlp.2.bias',
 'stage3.layers.2.0.layer_norm2.weight',
 'stage3.layers.2.0.layer_norm2.bias',
 'stage3.layers.2.1.layer_norm1.weight',
 'stage3.layers.2.1.layer_norm1.bias',
 'stage3.layers.2.1.window_attn.ref_tab',
 'stage3.layers.2.1.window_attn.rel_pos_1D',
 'stage3.layers.2.1.window_attn.qkv.weight',
 'stage3.layers.2.1.window_attn.to_out.weight',
 'stage3.layers.2.1.window_attn.to_out.bias',
 'stage3.layers.2.1.mlp.0.weight',
 'stage3.layers.2.1.mlp.0.bias',
 'stage3.layers.2.1.mlp.2.weight',
 'stage3.layers.2.1.mlp.2.bias',
 'stage3.layers.2.1.layer_norm2.weight',
 'stage3.layers.2.1.layer_norm2.bias',
 'stage4.down_scale.linear.weight',
 'stage4.layers.0.0.layer_norm1.weight',
 'stage4.layers.0.0.layer_norm1.bias',
 'stage4.layers.0.0.window_attn.ref_tab',
 'stage4.layers.0.0.window_attn.rel_pos_1D',
 'stage4.layers.0.0.window_attn.qkv.weight',
 'stage4.layers.0.0.window_attn.to_out.weight',
 'stage4.layers.0.0.window_attn.to_out.bias',
 'stage4.layers.0.0.mlp.0.weight',
 'stage4.layers.0.0.mlp.0.bias',
 'stage4.layers.0.0.mlp.2.weight',
 'stage4.layers.0.0.mlp.2.bias',
 'stage4.layers.0.0.layer_norm2.weight',
 'stage4.layers.0.0.layer_norm2.bias',
 'stage4.layers.0.1.layer_norm1.weight',
 'stage4.layers.0.1.layer_norm1.bias',
 'stage4.layers.0.1.window_attn.ref_tab',
 'stage4.layers.0.1.window_attn.rel_pos_1D',
 'stage4.layers.0.1.window_attn.qkv.weight',
 'stage4.layers.0.1.window_attn.to_out.weight',
 'stage4.layers.0.1.window_attn.to_out.bias',
 'stage4.layers.0.1.mlp.0.weight',
 'stage4.layers.0.1.mlp.0.bias',
 'stage4.layers.0.1.mlp.2.weight',
 'stage4.layers.0.1.mlp.2.bias',
 'stage4.layers.0.1.layer_norm2.weight',
 'stage4.layers.0.1.layer_norm2.bias',
 'mlp.2.weight',
 'mlp.2.bias',
 'mlp.3.weight',
 'mlp.3.bias']