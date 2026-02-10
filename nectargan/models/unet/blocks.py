# Copyright 2025 Zachary Bork
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

from typing import Literal

import  torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True)
        
    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        x_flat = x_norm.reshape(b, c, h * w).transpose(1, 2)
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        attn_out = attn_out.transpose(1, 2).reshape(b, c, h, w)
        return x + attn_out

class UnetBlock(nn.Module):
    '''Defines a standard UNet block to be used by the generator model.'''
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            upconv_type: Literal['transposed', 'bilinear'], 
            activation: Literal[
                'leaky', 'relu', 'tanh', 'silu', 'linear'] | None,  
            norm: Literal['instance', 'group'] | None,
            num_groups: int=32,
            down: bool=True, 
            stride: int=2,
            bias: bool=True, 
            use_dropout: bool=False,
            dropout_chance: float=0.5,
            leaky_neg_slope: float=0.2,
            **kwargs
        ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.stride = stride
        modules = []
        if down:
            modules.append(nn.Conv2d(
                in_channels, self.out_channels, 
                kernel_size=4 if stride == 2 else 3, 
                stride=stride, padding=1, bias=bias, 
                padding_mode='reflect'))
        else:
            if stride == 2:
                match upconv_type.strip().casefold():
                    case 'transposed':
                        modules.append(nn.ConvTranspose2d(
                            in_channels, self.out_channels, 
                            kernel_size=4, stride=2, padding=1, bias=bias))
                    case 'bilinear':
                        modules.append(nn.Upsample(
                            scale_factor=2, 
                            mode='bilinear', 
                            align_corners=False))
                        modules.append(nn.ReflectionPad2d(1))
                        modules.append(nn.Conv2d(
                            in_channels, self.out_channels, 
                            kernel_size=3, stride=1, padding=0))
                    case _: raise ValueError('Invalid upsampling type.')
            else:
                modules.append(nn.ReflectionPad2d(1))
                modules.append(nn.Conv2d(
                    in_channels, self.out_channels, 
                    kernel_size=3, stride=1, padding=0, bias=bias))
        
        match norm:
            case 'instance': 
                modules.append(nn.InstanceNorm2d(self.out_channels))
            case 'group':
                modules.append(nn.GroupNorm(num_groups, self.out_channels))
            case None: modules.append(nn.Identity())
            case _: raise ValueError('Invalid normalization type.')

        match activation:
            case 'leaky': modules.append(nn.LeakyReLU(leaky_neg_slope))
            case 'relu': modules.append(nn.ReLU())
            case 'tanh': modules.append(nn.Tanh())
            case 'silu': modules.append(nn.SiLU())
            case 'linear' | None: modules.append(nn.Identity())
            case _: raise ValueError('Invalid activation function.')

        self.conv = nn.Sequential(*modules)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(dropout_chance)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class ResidualUnetBlock(UnetBlock):
    '''Defines a ResidualUNet block to be used by the generator model.'''
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int,
            upconv_type: Literal['transposed', 'bilinear'], 
            activation: Literal[
                'leaky', 'relu', 'tanh', 'silu', 'linear'] | None,  
            norm: Literal['instance', 'group'] | None, 
            residual_activation: Literal[
                'leaky', 'relu', 'tanh', 'silu', 'linear'] | None=None, 
            stride: int=2,  
            down: bool=True, 
            bias: bool=True, 
            use_dropout: bool=False,
            dropout_chance: float=0.5,
            **kwargs
        ) -> None:
        super().__init__(
            in_channels=in_channels, out_channels=out_channels,
            upconv_type=upconv_type, activation=activation, norm=norm, 
            down=down, stride=stride, bias=bias, use_dropout=use_dropout,
            dropout_chance=dropout_chance)
        
        match residual_activation:
            case 'leaky': _activation = nn.LeakyReLU(inplace=True)
            case 'relu':  _activation = nn.ReLU()
            case 'tanh':  _activation = nn.Tanh()
            case 'silu':  _activation = nn.SiLU()
            case 'linear' | None: _activation = nn.Identity()
            case _: raise ValueError('Invalid residual activation function.')
        
        modules = []
        if down:
            modules.append(nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=1, stride=stride))
            modules.append(_activation)
        else:
            if stride == 2:
                modules.append(nn.ConvTranspose2d(
                    in_channels, out_channels, 
                    kernel_size=1, stride=2, output_padding=1))
            else:
                modules.append(nn.Conv2d(
                    in_channels, out_channels, 
                    kernel_size=1, stride=1))
            modules.append(_activation)

        self.residual = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x) 
        out = self.dropout(out) if self.use_dropout else out
        return out + self.residual(x)

class TimeEmbeddedUnetBlock(ResidualUnetBlock):
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
        out = self.conv(x)
        if not self.time_embedding_dimension is None and not t is None:
            out = out + self.mlp(t).unsqueeze(-1).unsqueeze(-1) 
        out = self.dropout(out) if self.use_dropout else out
        return out + self.residual(x)

class CrossAttentionUnetBlock(TimeEmbeddedUnetBlock):
    def __init__(
            self, 
            num_heads: int=8,
            context_dimension: int=None,
            time_embedding_dimension: int=None, 
            min_attention_channels: int=32,
            attention_scaling_factor: float=1.0,
            **kwargs
        ) -> None:
        super().__init__(time_embedding_dimension, **kwargs)
        self.context_dim = context_dimension
        self.num_heads = num_heads
        attn_dim = self.out_channels
        self.attention_enabled = attn_dim > min_attention_channels
        self.attention_scale = nn.Parameter(
                torch.ones(1) * attention_scaling_factor)

        if self.attention_enabled:
            self.attention = nn.MultiheadAttention(
                embed_dim=attn_dim, 
                kdim=context_dimension,
                vdim=context_dimension,
                num_heads=num_heads, 
                batch_first=True)
            self.proj = nn.Linear(attn_dim, attn_dim)
            self.norm = nn.GroupNorm(32, self.out_channels)

    def forward(
            self, 
            x: torch.Tensor, 
            t: torch.Tensor, 
            context: torch.Tensor | None=None
        ) -> torch.Tensor:
        x = super().forward(x, t)
        if context is None or not self.attention_enabled: return x

        x_norm = self.norm(x)
        B, C, H, W = x_norm.shape
        x_flat = x_norm.view(B, C, H * W).permute(0, 2, 1)

        attn_out, _ = self.attention(x_flat, context, context)
        attn_out = self.proj(attn_out)

        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        return x + self.attention_scale * attn_out



