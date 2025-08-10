# %%
import torch as t
import torch.nn as nn

class mha(nn.Module):
    def __init__(self, dim: int, num_heads: int, 
                    qkv_bias: bool, att_drop: float, 
                    proj_drop: float) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3*dim, bias=qkv_bias)
        self.att_d = nn.Dropout(att_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_d = nn.Dropout(proj_drop)

    def forward(self, x: t.Tensor) -> t.Tensor:
        n_sample, patches, dim = x.shape
        if dim != self.dim:
            raise ValueError

        qkv: t.Tensor = self.qkv(x) # n_samples, patches, 3*dim
        qkv: t.Tensor = qkv.reshape(n_sample, patches, 3, self.num_heads, self.head_dim)
        qkv: t.Tensor = qkv.permute(2, 0, 3, 1, 4) # qkv, n_samples, num_heads, patches, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        weights: t.Tensor = q @ k.transpose(-2,-1) # n_samples, num_heads, num_patches, num_patches
        droped_weights: t.Tensor = self.att_d(weights.softmax(-1))
        weighted_value: t.Tensor = droped_weights @ v # n_samples, num_heads, num_patches, head_dim
        v_t: t.Tensor = weighted_value.transpose(1,2) # n_samples, num_patches, num_heads, head_dim
        v_flatten: t.Tensor = v_t.flatten(2) # n_samples, num_patches, dimension
        final_proj: t.Tensor = self.proj_d(self.proj(v_flatten))

        return x

# %%

if __name__ == '__main__':
    x: t.Tensor = t.randn(8, 16, 24)
    mha_c: mha = mha(24, 6, True, 0.2, 0.2)
    y: t.Tensor = mha_c(x)
    print(y.shape)