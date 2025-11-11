# Copyright 2025 Zachary Bork
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import torch
import torch.nn as nn

from nectargan.models.resnet.blocks import ResnetBlock

class ResNet(nn.Module):
    def __init__(
            self, 
            in_channels: int=3, 
            features: int=64, 
            block_type: nn.Module=ResnetBlock,
        ) -> None:
        super(ResNet, self).__init__()

        self.in_channels = in_channels
        self.features = features
        self.block_type = block_type

    def model(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            'This method is implemented by the child class.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)



