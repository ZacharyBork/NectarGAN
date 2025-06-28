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
[1, 512, 4, 4] ---> bottleneck --> [1, 512, 2, 2] → → → → up1 -> [1, 512, 4, 4]
'''
import torch
import torch.nn as nn
import torch.nn.init as init

from nectargan.models.unet.blocks import UnetBlock

class UnetGenerator(nn.Module):
    '''Defines a modular UNet style generator with a configurable layer count.

    Args:
        input_size : The resolution (^2) of the training input images.
        in_channels : The number of input channels in the training input images
            (i.e. 1 for mono, 3 for RGB).
        features : The number of features on the first downsampling layer.
        n_downs : The number of downsampling layers to add the the model.
        use_dropout_layers : The number of upsampling layers (starting from the
            deepest) to apply dropout on.
        block_type : What UNet block type to use when assembling the model.
        upconv_type : What upsampling type to use (i.e. 'Transposed', 
            'bilinear').
    '''
    def __init__(
            self, 
            input_size: int, 
            in_channels: int=3, 
            features: int=64, 
            n_downs: int=6, 
            use_dropout_layers: int=3, 
            block_type=UnetBlock, 
            upconv_type: str='Transposed'
        ) -> None:
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.features = features
        self.use_dropout_layers = use_dropout_layers
        self.n_down = n_downs
        self.block_type = block_type
        self.upconv_type = upconv_type

        self.build_model() # Initialize generator model

    def build_model(self) -> None:
        '''Wrapper function to assemble a full UNet generator model.'''
        self.validate_layer_count()       # Validate layer count for in shape
        self.build_channel_map()          # Build channel map
        self.define_downsampling_blocks() # Define downsampling blocks
        self.define_bottleneck()          # Define bottleneck
        self.define_upsampling_blocks()   # Define Upsampling Blocks
        self.apply(self.init_weights)     # Initialize layer weights

    def validate_layer_count(self) -> None:
        '''Checks number of downsampling layers against input image
        resolution to ensure that bottleneck channel size never reaches 0x0.

        ValueError if crop_size ≤ 2^(n_down+2)
        '''
        # Get shape of dummy image input
        shape = torch.randn(1, 3, self.input_size, self.input_size).shape[1:]
        
        # Define min resolution based on number of downsampling layers
        # +1 for initial_down, +1 for bottleneck
        min_size = 2 ** (self.n_down + 2)  

        # Check tensor shape against min resolution
        if any(s < min_size for s in shape[-2:]):
            # Raise error if input size is too small
            e = 'Input too small for n_down={}. Min size: {}x{}'
            raise ValueError(e.format(self.n_down, min_size, min_size)) 

    def build_channel_map(self) -> None:
        '''Assembles Unet channel structure.'''
        down_channels = [(self.in_channels, self.features)]
        in_features = out_features = self.features
        for i in range(self.n_down): # Create down layer IO shapes
            in_features = min(out_features, self.features*8)
            out_features = min(out_features*2, self.features*8)
            down_channels.append((in_features, out_features))

        # Create layer IO shapes skip channels
        skip_channels = list(reversed(
            [(out_ch, in_ch) for (in_ch, out_ch) in down_channels]))
        
        # Build IO shapes for ups with skips
        up_channels = [(in_ch*2, out_ch) for in_ch, out_ch in skip_channels] 

        # Add IO shape for first up layer
        up_channels.insert(0, (self.features*8, self.features*8)) 

        self.channel_map = {
            'initial_down': down_channels[0],
            'downs': down_channels[1:],
            'bottleneck': (self.features*8, self.features*8),
            'ups': up_channels[:-1],
            'final_up': up_channels[-1]
        }
    
    def define_downsampling_blocks(self) -> None:
        '''Defines the layers in the downsampling path.'''
        # Define initial downsampling layer
        self.initial_down = self.block_type(
            self.channel_map['initial_down'][0], 
            self.channel_map['initial_down'][1], 
            upconv_type=self.upconv_type, activation='leaky',
            norm=None, down=True, bias=True, use_dropout=False)

        # Define additional downsampling layers
        self.downs = nn.ModuleList()
        for (in_ch, out_ch) in self.channel_map['downs']:
            self.downs.append(
                self.block_type(
                    in_ch, out_ch, 
                    upconv_type=self.upconv_type, activation='leaky',
                    norm='instance', down=True, bias=False, use_dropout=False))

    def define_bottleneck(self) -> None:
        '''Defines the bottleneck layer.'''
        # Define bottleneck
        self.bottleneck = self.block_type(
            self.channel_map['bottleneck'][0], 
            self.channel_map['bottleneck'][1], 
            upconv_type=self.upconv_type, activation='relu',
            norm=None, down=True, bias=True, use_dropout=False)

    def define_upsampling_blocks(self) -> None:
        '''Defines the layers in the upsampling path.'''
        # Define upsampling layers
        self.ups = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(self.channel_map['ups']):
            self.ups.append(self.block_type(
                in_ch, out_ch, upconv_type=self.upconv_type, 
                activation='relu', norm='instance', down=False, bias=False, 
                use_dropout=i<self.use_dropout_layers))

        # Define final upsampling layer
        self.final_up = self.block_type(
            self.channel_map['final_up'][0], self.channel_map['final_up'][1], 
            upconv_type=self.upconv_type, activation='tanh',
            norm=None, down=False, bias=True, use_dropout=False)

    def init_weights(self, m: nn.Module) -> None:
        '''Initializes layer weights based on layer type.
        
        Args:
            m : The module to initialize weights for.
        '''
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d)):
            if m.weight is not None:
                init.normal_(m.weight, 1.0, 0.02)
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward function for UnetGenerator class.
        
        Args:
            x : The input tensor to run the generator's inference on.
        '''
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
    model = UnetGenerator()
    output = model(x)
    print(output.shape)
