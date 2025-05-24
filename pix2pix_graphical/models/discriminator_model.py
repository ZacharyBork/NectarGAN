import torch
import torch.nn as nn
import torch.nn.functional as F

from .discriminator_blocks import CNNBlock

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, n_layers=3, max_channels=512):
        super().__init__()

        layers = []

        # Initial conv layer
        layers.append(nn.Conv2d(in_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, padding_mode='reflect'))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # n_layers loop
        in_ch = base_channels
        for i in range(1, n_layers):
            out_ch = min(in_ch * 2, max_channels)
            stride = 1 if i == n_layers - 1 else 2
            layers.append(CNNBlock(in_ch, out_ch, stride=stride))
            in_ch = out_ch

        # Final output layer
        layers.append(nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'))

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.model(x)

if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator()
    output = model(x, y)
    print(output.shape)