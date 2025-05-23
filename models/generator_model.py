'''
                UNet-esque generator architecture
------------------------------------------------------------------------
Input: [1, 3, 512, 512] (512^2 RGB)                     Output:  [1, 3, 512, 512]
                        ↓                                 ↑
  |Input Shape|    |Down Layer|    |Output/Skip Shape| |Up Layer|  |Output Shape|
                        ↓                                 ↑
[1, 3, 512, 512] ---> init_down -> [1, 64, 256, 256] --> final_up --> [1, 3, 512, 512]
                        ↓                                 ↑
[1, 64, 256, 256] --> down1 -----> [1, 128, 128, 128] -> up7 -------> [1, 64, 256, 256]
                        ↓                                 ↑
[1, 128, 128, 128] -> down2 -----> [1, 256, 64, 64] ---> up6 -------> [1, 128, 128, 128]
                        ↓                                 ↑
[1, 256, 64, 64] ---> down3 -----> [1, 512, 32, 32] ---> up5 -------> [1, 256, 64, 64]
                        ↓                                 ↑
[1, 512, 32, 32] ---> down4 -----> [1, 512, 16, 16] ---> up4 -------> [1, 512, 32, 32]
                        ↓                                 ↑
[1, 512, 16, 16] ---> down5 -----> [1, 512, 8, 8] -----> up3 -------> [1, 512, 16, 16]
                        ↓                                 ↑ 
[1, 512, 8, 8] -----> down6 -----> [1, 512, 4, 4] -----> up2 -------> [1, 512, 8, 8]
                        ↓                                 ↑
                        ↓     ← ← ← extra layers → → →    ↑
                        ↓                                 ↑
[1, 512, 4, 4] ---> bottleneck --> [1, 512, 2, 2] → → → → up1 -> [1, 512, 4, 4]
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .generator_blocks import UnetBlock
from .generator_blocks import ResidualUnetBlock

class Generator(nn.Module):
    '''This class defines a modular UNet style generator with a configurable layer count.'''
    def __init__(self, input_size: int, in_channels: int=3, 
                 features: int=64, extra_layers: int=0, 
                 block_type=UnetBlock, upconv_type: str='Transpose'):
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.features = features
        self.extra_layers = extra_layers
        self.n_down = 6
        self.block_type = block_type

        # Validate layer count for current input shape
        self.validate_layer_count()

        # Build channel map
        self.channel_map = self.build_channel_map()

        # Define initial downsampling layer
        self.initial_down = nn.Sequential(
            nn.Conv2d(
                self.channel_map['initial_down'][0], self.channel_map['initial_down'][1], 
                kernel_size=4, stride=2, padding=1, padding_mode='reflect'), 
            nn.LeakyReLU(0.2),   
        )

        # Define additional downsampling layers
        self.downs = nn.ModuleList()
        for (in_ch, out_ch) in self.channel_map['downs']:
            self.downs.append(block_type(in_ch, out_ch, down=True, use_dropout=False))

        # Define bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                self.channel_map['bottleneck'][0], self.channel_map['bottleneck'][1], 
                kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.ReLU(),
        )

        # Define upsampling layers
        self.ups = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(self.channel_map['ups']):
            self.ups.append(block_type(in_ch, out_ch, down=False, use_dropout=i<3))

        # Define final upsampling layer
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(
                self.channel_map['final_up'][0], self.channel_map['final_up'][1], 
                kernel_size=4, stride=2, padding=1),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # nn.Conv2d(
            #     self.channel_map['final_up'][0], self.channel_map['final_up'][1], 
            #     kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

        # Initialize layer weights
        self.apply(self.init_weights)


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

    def build_channel_map(self):
        '''Assembles Unet channel structure.'''
        down_channels = [(self.in_channels, self.features)]
        in_features = out_features = self.features
        for i in range(self.n_down): # Create down layer IO shapes
            in_features = min(out_features, self.features*8)
            out_features = min(out_features*2, self.features*8)
            down_channels.append((in_features, out_features))

        # Create extra layer IO shapes and add to down channels list
        extra_layers = [(self.features*8, self.features*8) for _ in range(self.extra_layers)]
        down_channels = down_channels[:self.n_down+1] + extra_layers

        # Create layer IO shapes skip channels
        skip_channels = list(reversed([(out_ch, in_ch) for (in_ch, out_ch) in down_channels]))
        
        # Build IO shapes for ups with skips
        up_channels = [(in_ch*2, out_ch) for in_ch, out_ch in skip_channels] 

        # Add IO shape for first up layer
        up_channels.insert(0, (self.features*8, self.features*8)) 

        return {
            'initial_down': down_channels[0],
            'downs': down_channels[1:],
            'bottleneck': (self.features*8, self.features*8),
            'ups': up_channels[:-1],
            'final_up': up_channels[-1]
        }
    
    def init_weights(self, m):
        '''Initializes layer weights based on layer type.'''
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d)):
            if m.weight is not None:
                init.normal_(m.weight, 1.0, 0.02)
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial_down(x) # Run downsampling layer
        skips = [x] # Store outputs for skip connections
        for down in self.downs:
            x = down(x)
            skips.append(x)

        skips.reverse() # Align skips with up conv layer
        x = self.bottleneck(x) # Run bottleneck layer

        for i, up in enumerate(self.ups):
            # No skip connection on first upconv layer
            if i == 0: x = up(x)
            else: # Stack x and skip then upsample
                skip = skips[i-1]
                x = up(torch.cat([x, skip], dim=1))

        # Return result of final upsampling layer
        return self.final_up(torch.cat([x, skips[-1]], dim=1))

if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    model = Generator()
    output = model(x)
    print(output.shape)
