from parts import Stages
from config import Config

import torch as t
from typing import Tuple

class swin_transformer(t.nn.Module):
    def __init__(self, in_channesl: int, hidden_dim: int, layers:Tuple[int, ...], heads: Tuple[int, ...], num_clas: int,
                    head_dim: int, window_size: int, downscaling_factor: Tuple[int,...], relative_position: bool = True):
        super().__init__()
        self.stage1 = Stages(in_dim=in_channesl, hidden_dim=hidden_dim, layers=layers[0], downscaling_factor=downscaling_factor[0],
                                heads=heads[0], head_dim=head_dim, window_size=window_size, rel_pos=relative_position)
        self.stage2 = Stages(in_channesl=hidden_dim, hidden_dim=hidden_dim*2, layers=layers[1], downscaling_factor=downscaling_factor[1],
                                heads=heads[1], head_dim=head_dim, window_size=window_size, rel_pos=relative_position)
        self.stage3 = Stages(in_channesl=in_channesl*2, hidden_dim=hidden_dim*4, layers=layers[2], downscaling_factor=downscaling_factor[2],
                                heads=heads[2], head_dim=head_dim, window_size=window_size, rel_pos=relative_position)
        self.stage4 = Stages(in_channesl=in_channesl*4, hidden_dim=hidden_dim*8, layers=layers[3], downscaling_factor=downscaling_factor[3],
                                heads=heads[3], head_dim=head_dim, window_size=window_size, rel_pos=relative_position)
        self.mlp = t.nn.Sequential(
            t.nn.AdaptiveAvgPool2d(output_size=1),
            t.nn.Linear(in_features=hidden_dim*8, out_features=num_clas)
        )

    def forward(self, x:t.Tensor) -> t.Tensor:
        return self.mlp(self.stage4(self.stage3(self.stage2(self.stage1(x)))))