import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import UnetBlock

class Generator(nn.Module):
    def __init__(self, in_channels: int=3, features: int=64, n_up: int=7, n_down: int=6):
        super().__init__()
        self.in_channels = in_channels
        self.features = features
        self.n_down = n_down
        self.n_up = n_up

        down_channels, up_channels, final_up_in_ch, final_down_out = self.build_unet_channels()

        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )

        self.downs = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(down_channels):
            self.downs.append(UnetBlock(in_ch, out_ch, down=True, act='leaky', use_dropout=False))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(final_down_out, final_down_out, 4, 2, 1, padding_mode='reflect'),
            nn.ReLU()
        )

        self.ups = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(up_channels):
            use_dropout = i < 3  # Dropout on first 3 up blocks
            self.ups.append(UnetBlock(in_ch, out_ch, down=False, act='relu', use_dropout=use_dropout))

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(final_up_in_ch, in_channels, 4, 2, 1),
            nn.Tanh()
        )

    def build_unet_channels(self, max_channels=512):
        down_channels = []
        in_ch = self.features

        for _ in range(self.n_down):
            out_ch = min(in_ch * 2, max_channels)
            down_channels.append((in_ch, out_ch))
            in_ch = out_ch
        final_down_out = in_ch
        
        up_channels = []
        skip_channels = [out_ch for (_, out_ch) in down_channels[:-1]][::-1]
        in_ch = down_channels[-1][1]

        for skip_ch in skip_channels:
            up_in_ch = in_ch + skip_ch
            up_out_ch = skip_ch
            up_channels.append((up_in_ch, up_out_ch))
            in_ch = up_out_ch

        final_up_in_ch = in_ch + down_channels[0][0]

        return down_channels, up_channels, final_up_in_ch, final_down_out

    def forward(self, x):
        x = self.initial_down(x)
        skips = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = self.bottleneck(x)
        skips = list(reversed(skips[:-1]))
        
        for i, up in enumerate(self.ups):
            skip = skips[i]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = up(torch.cat([x, skip], dim=1))

        final_skip = skips[-1]
        if x.shape[2:] != final_skip.shape[2:]:
            x = F.interpolate(x, size=final_skip.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, final_skip], dim=1)
        return self.final_up(x)

def test():
    x = torch.randn((1, 3, 256, 256))
    model = Generator(in_channels=3, features=64)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()