import torch as t
from MHA import mha

import configurations

info = configurations.config()
class PatchEmbeddings(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embd = t.nn.Conv2d(in_channels=info.num_channels, 
                                        out_channels=info.embedding_dim, stride=info.patch_size,
                                        kernel_size=info.patch_size)

    def forward(self, x:t.Tensor) -> t.Tensor:
        return t.flatten(self.patch_embd(x), 2).transpose(1,2)
    
class TransformerEncoder(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_norm1 = t.nn.LayerNorm(info.embedding_dim)
        self.MHA = t.nn.MultiheadAttention(embed_dim=info.embedding_dim, 
                                            num_heads=info.attention_head, batch_first=True)
        # self.MHA = mha(dim=info.embedding_dim, num_heads=info.attention_head, qkv_bias=True,
        #                 att_drop=0.3, proj_drop=0.2)
        self.layer_norm2 = t.nn.LayerNorm(info.embedding_dim)
        self.mlp = t.nn.Sequential(
            t.nn.Linear(info.embedding_dim, info.mlp_nodes),
            t.nn.GELU(),
            t.nn.Linear(info.mlp_nodes, info.embedding_dim)
        )

    def forward(self, x:t.Tensor) -> t.Tensor:
        res1 = x
        x = self.MHA(self.layer_norm1(x), self.layer_norm1(x), self.layer_norm1(x))[0] + res1
        # x = self.MHA(self.layer_norm1(x)) + res1
        res2 = x
        x = self.mlp(self.layer_norm2(x)) + res2
        return x

class MLP(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = t.nn.LayerNorm(info.embedding_dim)
        self.linear_layers = t.nn.Sequential(
            t.nn.Linear(info.embedding_dim, info.mlp_nodes),
            t.nn.ReLU(),
            t.nn.Linear(info.mlp_nodes, info.num_classes)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.linear_layers(self.norm(x))
        return x

class ViT(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.get_embed = PatchEmbeddings()
        self.cls = t.nn.Parameter(t.randn(1, 1, info.embedding_dim))
        self.pos_emb = t.nn.Parameter(t.randn(1, info.patch_num+1, info.embedding_dim))
        self.transformer_block = t.nn.Sequential(*[TransformerEncoder() for _ in range(info.transformer_block)])
        self.mlp = MLP()

    def forward(self, x:t.Tensor) -> t.Tensor:
        x = self.get_embed(x)
        B = x.shape[0]
        cls_tkn = self.cls.expand(B,-1,-1)
        x = t.concat((cls_tkn, x), dim=1)
        x = x + self.pos_emb
        x = self.transformer_block(x)[:,0]
        x = self.mlp(x)
        return x
