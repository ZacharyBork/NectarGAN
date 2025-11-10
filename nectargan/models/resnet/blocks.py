# Copyright 2025 Zachary Bork
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            identity_downsample=None, 
            stride=1
        ) -> None:
        super(ResnetBlock, self).__init__()
        
        self.expansion = 4
        self.relu = nn.ReLU()

        self.conv = nn.Sequential(
            *self._build_layers(in_channels, out_channels, stride))
        self.identity_downsample = identity_downsample

    def _build_layers(
            self, 
            in_channels: int, 
            out_channels: int,
            stride: int
        ) -> list[nn.Module]:
        layers = []
        in_ch = in_channels
        for i in range(3): 
            out_ch = out_channels * max(1, self.expansion * (i-1))
            layers.append(
                nn.Conv2d(
                    in_ch, out_ch,
                    kernel_size=[1, 3][i%2], # 1x1 ->  3x3   -> 1x1
                    stride=[1, stride][i%2], #  1  -> stride -> 1
                    padding=[0, 1][i%2]))    #  0  ->   1    -> 0
            layers.append(nn.BatchNorm2d(out_ch))
            if i < 2: layers.append(self.relu)
            in_ch = out_channels
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv(x)

        identity = self.identity_downsample(identity) \
            if not self.identity_downsample is None else identity

        return self.relu(x + identity)