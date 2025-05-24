import torch
import torch.nn as nn
from typing import Union


class UnetBlock(nn.Module):
    '''This class defines a standard UNet block to be used by the generator model.'''
    def __init__(
            self, in_channels: int, out_channels: int, upconv_type: str, 
            activation: Union[str, None], norm: Union[str, None], down: bool=True, bias: bool=True, use_dropout: bool=False):
        super().__init__()
        modules = []
        if down:
            modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=bias, padding_mode='reflect'))
        else:
            match upconv_type:
                case 'Transpose':
                    modules.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=bias))
                case 'Bilinear':
                    modules.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
                    modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
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

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x
    
class ResidualUnetBlock(nn.Module):
    '''This class defines a ResidualUNet block to be used by the generator model.'''
    def __init__(self, in_channels, out_channels, down=True, act='relu', use_dropout=False):
        super().__init__()
        self.down = down
        self.use_dropout = use_dropout
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Define current activation function
        self.activation = nn.ReLU() if act == 'relu' else nn.LeakyReLU(0.2)

        # Define normalization
        norm = nn.InstanceNorm2d

        if down: # Downsampling layer
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
                norm(out_channels),
                self.activation,
            )
        else: # Upsampling layer
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                norm(out_channels),
                self.activation,
            )

        if in_channels != out_channels or down: # Residual shortcut
            # This will almost always be a 1x1 conv with stride=2 except at 
            # the bottleneck or if # of features exceeds the cap (Generator.features * 8)
            if down: self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
            else: self.residual = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=2, output_padding=1)
        else: self.residual = nn.Identity() # Residual = Identity if we hit feature cap 

        if use_dropout: # Define dropout if applicable
            self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Add conv output and residual
        out = self.conv(x) + self.residual(x)
        # Apply dropout if applicable
        return self.dropout(out) if self.use_dropout else out