import torch
import torch.nn as nn

from nectargan.models.unet.blocks import UnetBlock

class TimeEmbeddedUnetBlock(UnetBlock):
    def __init__(
            self, 
            time_embedding_dimension: int=None,
            **kwargs
        ) -> None:
        super().__init__(**kwargs)
        self.time_embedding_dimension = time_embedding_dimension
        if self.time_embedding_dimension is not None:
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embedding_dimension, self.out_channels))

    def forward(
            self, 
            x: torch.Tensor, 
            t: torch.Tensor
        ) -> torch.Tensor:
        x = self.conv(x)

        if not self.time_embedding_dimension is None and not t is None:
            x = x + self.mlp(t).unsqueeze(-1).unsqueeze(-1) 
        
        return self.dropout(x) if self.use_dropout else x
    
