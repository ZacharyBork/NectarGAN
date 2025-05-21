import torch
import torch.nn as nn
import torch.nn.functional as F

from .generator_blocks import UnetBlock
from .generator_blocks import ResidualUnetBlock

class Generator(nn.Module):
    def __init__(self, input_size: int, in_channels: int=3, features: int=64, n_down: int=6, block_type=UnetBlock):
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.features = features
        self.n_down = n_down
        self.n_up = self.n_down+1

        self.block_type = block_type

        # Validate layer count for current input shape
        self.validate_layer_count()

        # Generate channel structure
        down_channels, bottleneck_channels, up_channels, final_up_channels = self.build_unet_channels()

        # Define initial downsampling layer
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )

        # Define additional downsampling layers
        self.downs = nn.ModuleList()
        for (in_ch, out_ch) in down_channels:
            self.downs.append(block_type(in_ch, out_ch, down=True, act='leaky', use_dropout=False))

        # Define bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.ReLU(),
        )

        # Define upsampling layers
        self.ups = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(up_channels):
            self.ups.append(block_type(in_ch, out_ch, down=False, act='relu', use_dropout=i<3))

        # Define final upsampling layer
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(final_up_channels, in_channels, 4, 2, 1),
            nn.Tanh()
        )

    def validate_layer_count(self):
        '''Checks number of downsampling layers against input image
        resolution to ensure that bottleneck channel size never reaches 0x0.

        ValueError if crop_size ≤ 2^(n_down+2)
        '''
        # Get shape of dummy image input
        shape = torch.randn(1, 3, self.input_size, self.input_size).shape[1:]
        # Define min resolution based on number of downsampling layers
        min_size = 2 ** (self.n_down + 2)  # +1 for initial_down, +1 for bottleneck
        if any(s < min_size for s in shape[-2:]): # Check tensort shape against min resolution
            e = f'Input too small for n_down={self.n_down}. Min size: {min_size}x{min_size}'
            raise ValueError(e) # Raise error if input images are too small

    def build_unet_channels(self):
        '''Assembles Unet channel structure.'''
        in_ch = self.features
        down_channels = [] # Build downsampling layers features list
        for _ in range(self.n_down):
            out_ch = min(in_ch * 2, self.features * 8)
            down_channels.append((in_ch, out_ch))
            in_ch = out_ch
        bottleneck_channels = in_ch
        
        # Skip channels: reverse(down_channels - bottlneck_channels)
        skip_channels = [out_ch for (_, out_ch) in down_channels[:-1]][::-1]

        # Get output of last down block
        in_ch = down_channels[-1][1]

        up_channels = [] # Build upsampling layers features list
        for skip_ch in skip_channels:
            up_in_ch = in_ch + skip_ch
            up_out_ch = skip_ch
            up_channels.append((up_in_ch, up_out_ch))
            in_ch = up_out_ch

        final_up_channels = in_ch + down_channels[0][0]

        return down_channels, bottleneck_channels, up_channels, final_up_channels

    def forward(self, x):
        # Run downsampling layer
        x = self.initial_down(x)
        skips = [x] # Store outputs for skip connections
        for down in self.downs:
            x = down(x) 
            skips.append(x) 

        # Run bottleneck layer
        x = self.bottleneck(x)

        # Align skip connections
        skips = list(reversed(skips[:-1]))
        
        # Upsample with skip connections
        for i, up in enumerate(self.ups):
            skip = skips[i]
            if x.shape[2:] != skip.shape[2:]:
                # Ensure shape match between up layer and skip connection
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            # Stack previous up layer result and skip, then upsample
            x = up(torch.cat([x, skip], dim=1))

        # Same process for final up
        # Get skip, match shape, stack, upsample, return result
        final_skip = skips[-1]
        if x.shape[2:] != final_skip.shape[2:]:
            x = F.interpolate(x, size=final_skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, final_skip], dim=1)
        return self.final_up(x)

if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    model = Generator()
    output = model(x)
    print(output.shape)