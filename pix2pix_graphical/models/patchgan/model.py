import torch
import torch.nn as nn
import torch.nn.functional as F

from pix2pix_graphical.models.patchgan.blocks import CNNBlock

class Discriminator(nn.Module):
    '''Defines a PatchGAN discriminator model with a configurable layer count.
    '''
    def __init__(
            self, 
            in_channels=3, 
            base_channels=64, 
            n_layers=3, 
            max_channels=512
        ) -> None:
        super().__init__()

        self.layers = []

        # Add initial conv layer
        self.add_initial_layer(in_channels, base_channels)

        # Add n_layers hidden layers
        self.in_ch = base_channels
        self.add_n_layers(n_layers, max_channels)

        # Add final output layer
        self.add_final_layer()
        
        # Build network from layers
        self.model = nn.Sequential(*self.layers)

    def add_initial_layer(self, in_channels: int, base_channels: int) -> None:
        '''Defines the initial downsampling layer.

        Args:
            in_channels : Number of input channels for first conv layer.
            base_channels : Number of output channels for first conv layer.
        '''
        self.layers.append(
            nn.Conv2d(
                in_channels * 2, 
                base_channels, 
                kernel_size=4, stride=2, 
                padding=1, padding_mode='reflect'))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))

    def add_n_layers(
            self, 
            n_layers: int,
            max_channels: int
        ) -> None:
        '''Defines n_layers additional conv layers for discriminator.

        Args:
            n_layers : number of additional layers to construct.
            max_channels : Maximum number of input/output channels that any 
                created conv layer is allowed to have.
        '''
        for i in range(1, n_layers):
            out_ch = min(self.in_ch * 2, max_channels)
            stride = 1 if i == n_layers - 1 else 2
            self.layers.append(CNNBlock(self.in_ch, out_ch, stride=stride))
            self.in_ch = out_ch

    def add_final_layer(self) -> None:
        '''Defines final output layer of discriminator.

        This final layer reduces the channel count to 1, 1=real or 0=fake, or 
        some value in between depending on discriminator confidence.

        Args:
            in_ch: Input channel count for final conv layer.
        '''
        self.layers.append(
            nn.Conv2d(
                self.in_ch, 1, 
                kernel_size=4, stride=1, 
                padding=1, padding_mode='reflect'))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        '''Forward function for PatchGan discriminator.
        
        Discriminator infers on (x, y) pair and returns prediction.

        Args:
            x : Input image torch.Tensor.
            y : Either real ground truth, or generator fake as torch.Tensor.

        Returns:
            torch.Tensor : The discriminator's prediction on the input pair.
        '''
        x = torch.cat([x, y], dim=1)
        return self.model(x)

if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator()
    output = model(x, y)
    print(output.shape)