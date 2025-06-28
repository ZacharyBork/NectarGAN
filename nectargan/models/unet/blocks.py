import copy
from typing import Union

import  torch
import torch.nn as nn

class UnetBlock(nn.Module):
    '''Defines a standard UNet block to be used by the generator model.'''
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            upconv_type: str, 
            activation: Union[str, None], 
            norm: Union[str, None], 
            down: bool=True, 
            bias: bool=True, 
            use_dropout: bool=False
        ) -> None:
        super().__init__()
        modules = []
        if down:
            modules.append(
                nn.Conv2d(
                    in_channels, out_channels, 
                    kernel_size=4, stride=2, padding=1, 
                    bias=bias, padding_mode='reflect'))
        else:
            match upconv_type:
                case 'Transposed':
                    modules.append(nn.ConvTranspose2d(
                        in_channels, out_channels, 
                        kernel_size=4, stride=2, padding=1, bias=bias))
                case 'Bilinear':
                    modules.append(nn.Upsample(
                        scale_factor=2, 
                        mode='bilinear', 
                        align_corners=False))
                    modules.append(nn.ReflectionPad2d(1))
                    modules.append(nn.Conv2d(
                        in_channels, out_channels, 
                        kernel_size=3, stride=1, padding=0))
                case _: raise ValueError('Invalid upsampling type.')
        
        match norm:
            case 'instance': modules.append(nn.InstanceNorm2d(out_channels))
            case None: pass
            case _: raise ValueError('Invalid normalization type.')

        match activation:
            case 'leaky': modules.append(nn.LeakyReLU(0.2))
            case 'relu': modules.append(nn.ReLU())
            case 'tanh': modules.append(nn.Tanh())
            case None: pass
            case _: raise ValueError('Invalid activation function.')

        self.conv = nn.Sequential(*modules)

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class ResidualUnetBlock(UnetBlock):
    '''Defines a ResidualUNet block to be used by the generator model.'''
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            upconv_type: str, 
            activation: Union[str, None], 
            norm: Union[str, None], 
            down: bool=True, 
            bias: bool=True, 
            use_dropout: bool=False
        ) -> None:
        super().__init__(
            in_channels=in_channels, out_channels=out_channels,
            upconv_type=upconv_type, activation=activation, norm=norm, 
            down=down, bias=bias, use_dropout=use_dropout)
        
        modules = []
        if in_channels != out_channels or down: # Residual shortcut
            # This will almost always be a 1x1 conv with stride=2 except at the 
            # bottleneck or if # of features exceeds cap (Generator.features*8)
            if down: 
                modules.append(nn.Conv2d(
                    in_channels, out_channels, 
                    kernel_size=1, stride=2))
            else: 
                modules.append(nn.ConvTranspose2d(
                    in_channels, out_channels, 
                    kernel_size=1, stride=2, output_padding=1))
            modules.append(nn.ReLU(inplace=True))
        else: modules.append(nn.Identity()) # Residual=Identity if we hit max 

        
        self.residual = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add conv output and residual
        x = self.conv(x) + self.residual(x)
        # Apply dropout if applicable
        return self.dropout(x) if self.use_dropout else x

