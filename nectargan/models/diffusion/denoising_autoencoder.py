import torch
import torch.nn as nn

from nectargan.models import UnetGenerator
from nectargan.models.diffusion.blocks import \
    TimeEmbeddedUnetBlock, CrossAttentionUnetBlock
from nectargan.config import DiffusionConfig

class UnetDAE(UnetGenerator):
    '''UNet-based diffusion autoencoder.'''
    def __init__(
            self, 
            config: DiffusionConfig,
            context_dimension: int | None=None,
            **kwargs
        ) -> None:
        self.config = config
        self.cfg_dae = self.config.model.dae
        self.device = config.common.device
        self.block_type = CrossAttentionUnetBlock
        self.context_dimension = context_dimension
        self.time_embedding_dim = self.cfg_dae.time_embedding_dimension
        self.mlp_hidden_dim = self.cfg_dae.mlp_hidden_dimension
        self.mlp_output_dim = self.cfg_dae.mlp_output_dimension
        super().__init__(
            in_channels=config.model.dae.in_channels,
            input_size=config.model.input_size,
            features=config.model.dae.features,
            n_downs=config.model.dae.n_downs,
            block_type=self.block_type, 
            **kwargs)
        
        self.get_embedding_frequency()
        self.init_mlp()
        self.apply(self.init_weights)

    def init_mlp(self) -> None:
        D = self.cfg_dae
        mlp_layers = [
            nn.Linear(self.time_embedding_dim, self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_output_dim)]
        self.mlp = nn.Sequential(*mlp_layers)

    def get_embedding_frequency(self) -> None:
        freq = torch.arange(
            0, self.time_embedding_dim, 2).float()
        freq /= self.time_embedding_dim
        self.embedding_freq = (1 / (10000 ** (freq))).to(self.device)

    def embed_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        args = timesteps.unsqueeze(-1) * self.embedding_freq.unsqueeze(0)
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(embeddings)

    ##### BLOCK METHOD OVERRIDES #####
    # This needs to be changed in the future. These block functions should just 
    # accept kwargs in the base UNet generator class. Right now, every child 
    # generator will basically need to redefine the entire network structure 
    # to encode any additional data which is dumb.

    def define_downsampling_blocks(self) -> None:
        '''Defines the layers in the downsampling path.'''
        # Define initial downsampling layer
        self.initial_down = self.block_type(
            in_channels=self.channel_map['initial_down'][0], 
            out_channels=self.channel_map['initial_down'][1], 
            upconv_type=self.upconv_type, activation='leaky',
            norm='group', down=True, bias=True, use_dropout=False,
            time_embedding_dimension=self.time_embedding_dim,
            context_dimension=self.context_dimension)

        # Define additional downsampling layers
        self.downs = nn.ModuleList()
        for (in_ch, out_ch) in self.channel_map['downs']:
            self.downs.append(
                self.block_type(
                    in_channels=in_ch, out_channels=out_ch, 
                    upconv_type=self.upconv_type, activation='leaky',
                    norm='group', down=True, bias=False, use_dropout=False,
                    time_embedding_dimension=self.time_embedding_dim,
                    context_dimension=self.context_dimension))

    def define_bottleneck(self) -> None:
        '''Defines the bottleneck layer.'''
        # Define bottleneck
        self.bottleneck = self.block_type(
            in_channels=self.channel_map['bottleneck'][0], 
            out_channels=self.channel_map['bottleneck'][1], 
            upconv_type=self.upconv_type, activation='relu', norm='group', 
            down=True, bias=True, use_dropout=False,
            time_embedding_dimension=self.time_embedding_dim,
            context_dimension=self.context_dimension)

    def define_upsampling_blocks(self) -> None:
        '''Defines the layers in the upsampling path.'''
        # Define upsampling layers
        self.ups = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(self.channel_map['ups']):
            self.ups.append(self.block_type(
                in_channels=in_ch, out_channels=out_ch, 
                upconv_type=self.upconv_type, 
                activation='relu', norm='group', down=False, bias=False, 
                use_dropout=i<self.use_dropout_layers,
                time_embedding_dimension=self.time_embedding_dim,
                context_dimension=self.context_dimension))

        # Define final upsampling layer
        self.final_up = self.block_type(
            in_channels=self.channel_map['final_up'][0], 
            out_channels=self.channel_map['final_up'][1],
            upconv_type=self.upconv_type, activation=None,
            norm=None, down=False, bias=True, use_dropout=False,
            time_embedding_dimension=self.time_embedding_dim,
            context_dimension=self.context_dimension)

    def forward(
            self, 
            x: torch.Tensor, 
            timesteps: torch.Tensor,
            context: torch.Tensor | None=None
        ) -> torch.Tensor:
        embed_t = self.embed_timesteps(timesteps)

        x = self.initial_down(x, embed_t, context)
        skips = [x]
        for down in self.downs:
            x = down(x, embed_t, context)
            skips.append(x)

        skips.reverse()
        x = self.bottleneck(x, embed_t, context)
        x = self.ups[0](x, embed_t, context)

        for i, up in enumerate(self.ups[1:]):
            skip = skips[i]
            x = up(torch.cat([x, skip], dim=1), embed_t, context)

        return self.final_up(
            torch.cat([x, skips[-1]], dim=1), embed_t, context)

if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)
    t = torch.tensor([500])

    unet_dae = UnetDAE(
        device='cpu',
        time_embedding_dimension=128,
        mlp_hidden_dimension=256,
        mlp_output_dimension=128)

    parameters = 0
    for p in unet_dae.parameters():
        if p.requires_grad: parameters += p.numel()
    print(f'Parameters: {parameters}')
    
    y = unet_dae(x, t)
    print(f'Output Shape: {y.shape}')


