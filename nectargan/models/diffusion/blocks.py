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
            t: torch.Tensor,
            context: torch.Tensor | None=None
        ) -> torch.Tensor:
        x = self.conv(x)

        if not self.time_embedding_dimension is None and not t is None:
            x = x + self.mlp(t).unsqueeze(-1).unsqueeze(-1) 
        
        return self.dropout(x) if self.use_dropout else x

class CrossAttentionUnetBlock(TimeEmbeddedUnetBlock):
    def __init__(
            self, 
            num_heads: int=8,
            context_dimension: int=None,
            time_embedding_dimension: int=None, 
            min_attention_channels: int=32,
            **kwargs
        ) -> None:
        super().__init__(time_embedding_dimension, **kwargs)
        self.context_dim = context_dimension
        self.num_heads = num_heads
        attn_dim = self.out_channels
        self.attention_enabled = attn_dim > min_attention_channels

        if self.attention_enabled:
            self.to_queries = nn.Linear(attn_dim, attn_dim)
            self.to_keys = nn.Linear(context_dimension, attn_dim)
            self.to_values = nn.Linear(context_dimension, attn_dim)
            self.attention = nn.MultiheadAttention(
                embed_dim=attn_dim, num_heads=num_heads, batch_first=True)
            self.proj = nn.Linear(attn_dim, attn_dim)

    def forward(
            self, 
            x: torch.Tensor, 
            t: torch.Tensor, 
            context: torch.Tensor | None=None
        ) -> torch.Tensor:
        x = super().forward(x, t)
        if context is None or not self.attention_enabled: return x

        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)

        attn_out, _ = self.attention(
            self.to_queries(x_flat), 
            self.to_keys(context), 
            self.to_values(context))
        attn_out = self.proj(attn_out)

        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        return x + attn_out


