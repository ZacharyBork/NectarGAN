# Copyright 2025 Zachary Bork
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

from typing import Literal

import torch
import torch.nn as nn

from nectargan.models.resnet.blocks import ResnetBlock

class ResNet(nn.Module):
    def __init__(
            self, 
            in_ch: int=3, 
            num_classes: int=1000,
            layer_count: Literal[50, 101, 152]=50,
            block: nn.Module=ResnetBlock
        ) -> None:
        super(ResNet, self).__init__()

        self._define_layers(layer_count)
        self.block_type = block
        
        self.in_channels = 64
        self.initial_layer = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def _define_layers(self, layer_count: int) -> None:
        match layer_count:
            case 50:  self.layers = [3, 4, 6, 3]
            case 101: self.layers = [3, 4, 23, 3]
            case 152: self.layers = [3, 8, 36, 3]
            case _: 
                raise ValueError(
                    f'Invalid layer count for ResNet: {layer_count}\n'
                    f'Valid values are [50, 101, 152]')

    def _make_layer(
            self, 
            num_residual_blocks: int, 
            out_channels: int,
            stride: int
        ) -> None:
        identity_downsample = None
        seq = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels*4, 
                    kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4))
            
        seq.append(
            self.block_type(
                self.in_channels, out_channels, 
                identity_downsample, stride))
        self.in_channels = out_channels * 4

        for _ in range(num_residual_blocks - 1):
            seq.append(self.block_type(self.in_channels, out_channels))
        
        return nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_layer(x)

        out_ch = 64 
        for i in range(4):            
            layer = self._make_layer(
                self.layers[i], out_channels=out_ch, stride=min(i+1, 2))
            x = layer(x)
            out_ch *= 2

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    counts = [50, 101, 152]
    for count in counts:
        net = ResNet(layer_count=count)
        y: torch.Tensor = net(x).to('cuda')

        if y.shape == torch.Size([2, 1000]):
            status = 'Passing'
        else: status = 'Failing'
        print(f'Resnet{count}: {status}')


