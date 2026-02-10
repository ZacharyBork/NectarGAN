# Copyright 2025 Zachary Bork
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import math

import torch
import torch.nn as nn

from nectargan.models import ResNet
from nectargan.models.resnet.blocks import ResnetBlock

class ResNetGenerator(ResNet):
    def __init__(
            self, 
            n_downs: int=3,
            n_residual_blocks: int=9,
            in_channels: int=3, 
            features: int=64, 
            block_type: nn.Module=ResnetBlock
        ) -> None:
        super(ResNetGenerator, self).__init__(
            in_channels=in_channels,
            features=features,
            block_type=block_type)
        
        self.out_channels = self.in_channels
        self.n_downs = n_downs
        self.n_residual_blocks = n_residual_blocks

        self.sequence = nn.Sequential(*self._build_model())

    def _initial_layer(self) -> list[nn.Module]:
        modules = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                self.in_channels, self.features, kernel_size=7, 
                stride= 1, padding=0),
            nn.InstanceNorm2d(self.features),
            nn.ReLU(inplace=True)]
        self.in_channels = self.features
        return modules

    def _sampler_layer(self, decoder: bool=False) -> list[nn.Module]:
        output_padding = int(decoder)
        modules = []
        for _ in range(self.n_downs):
            if decoder:
                out_ch = math.floor(self.in_channels / 2)
                modules.append(
                    nn.ConvTranspose2d(
                        self.in_channels, out_ch, kernel_size=3, 
                        stride=2, padding=1, output_padding=output_padding))
            else:
                out_ch = self.in_channels * 2
                modules.append(
                    nn.Conv2d(
                        self.in_channels, out_ch, 
                        kernel_size=3, stride=2, padding=1))
            modules += [nn.InstanceNorm2d(out_ch), nn.ReLU(inplace=True)]
            self.in_channels = out_ch
        return modules
    
    def _residual_bottleneck(self) -> list[nn.Module]:
        modules = [
            self.block_type(
                self.in_channels, 
                math.floor(self.in_channels / 4)
            ) for _ in range(self.n_residual_blocks)]
        return modules

    def _final_layer(self) -> list[nn.Module]:
        modules = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                self.in_channels, self.out_channels, 
                kernel_size=7, padding=0),
            nn.Tanh()]
        return modules

    def _build_model(self) -> list[nn.Module]:
        seq = self._initial_layer()
        seq += self._sampler_layer()
        seq += self._residual_bottleneck()
        seq += self._sampler_layer(decoder=True)
        seq += self._final_layer()
        return seq
    
    def model(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequence(x)

if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)
    net = ResNetGenerator()
    y: torch.Tensor = net(x)
    print(y.shape == x.shape)




