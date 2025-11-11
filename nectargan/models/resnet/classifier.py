# Copyright 2025 Zachary Bork
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

from typing import Literal

import torch
import torch.nn as nn

from nectargan.models.resnet.model import ResNet

class ResNetClassifier(ResNet):
    def __init__(
            self, 
            n_classes: int=1000,
            layer_count: Literal[50, 101, 152]=50
        ) -> None:
        super(ResNetClassifier, self).__init__()

        self.out_channels = self.features
        self.layer_count = layer_count
        self.n_classes = n_classes

        self._define_layers()
        self._build_initial_layer()

    def _define_layers(self) -> None:
        match self.layer_count:
            case 50:  self.layers = [3, 4, 6, 3]
            case 101: self.layers = [3, 4, 23, 3]
            case 152: self.layers = [3, 8, 36, 3]
            case _: 
                raise ValueError(
                    f'Invalid layer count for ResNetClassifier: '
                    f'{self.layer_count}\nValid values are [50, 101, 152]')

    def _build_initial_layer(self) -> None:
        self.initial_layer = nn.Sequential(
            nn.Conv2d(
                self.in_channels, self.features, kernel_size=7, 
                stride=2, padding=3, padding_mode='zeros'),
            nn.BatchNorm2d(self.features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.in_channels = self.features

    def _build_model(
            self, 
            num_residual_blocks: int, 
            out_channels: int,
            stride: int
        ) -> list[nn.Module]:
        seq = []
        identity_downsample = None
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
        return seq
    
    def _apply_classifier_head(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.AdaptiveAvgPool2d((1, 1))(x).reshape(x.shape[0], -1)
        return nn.Linear(512 * 4, self.n_classes)(x)        

    def model(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_layer(x)
        for i in range(len(self.layers)):            
            layer = nn.Sequential(
                *self._build_model(
                    self.layers[i], out_channels=self.out_channels, 
                    stride=min(i+1, 2)))
            x = layer(x)
            self.out_channels *= 2
        return self._apply_classifier_head(x)

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    counts = [50, 101, 152]
    for count in counts:
        net = ResNetClassifier(layer_count=count)
        y: torch.Tensor = net(x).to('cuda')

        if y.shape == torch.Size([2, 1000]):
            status = 'Passing'
        else: status = 'Failing'
        print(f'Resnet{count}: {status}')




