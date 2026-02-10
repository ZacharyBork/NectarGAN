import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from nectargan.models import UnetGenerator
from nectargan.models.unet.blocks import \
    TimeEmbeddedUnetBlock, AttentionBlock
from nectargan.config import DiffusionConfig

class DiffusionUnet(UnetGenerator):
    '''UNet-based diffusion autoencoder.'''
    def __init__(
            self, 
            config: DiffusionConfig,
            block_type: TimeEmbeddedUnetBlock,
            context_dimension: int | None=None,
            bottleneck_depth: int=6,
            use_attention: bool=True,
            use_checkpointing: bool=True,
            **kwargs
        ) -> None:
        '''Initialized a DiffusionUnet.
        
        Args:
            config : The DiffusionConfig to use for the UNet.
            block_type : The conv block type for the network to use.
            context_dimension : The context dimension for the UNet blocks, if
                using text conditioning.
            use_attention : Whether to enable the self attention mechanism on
                deeper layers of the UNet.
            use_checkpointing : Whether to use model checkpointing during
                training. Can help reduce memory overhead significantly.
            kwargs : Any additional keyword arguments to pass to the base
                UnetGenerator class.
        '''
        self.config  = config
        self.device  = config.common.device
        self.cfg_unet = config.model.unet
        self.cfg_mlp = config.model.mlp
        
        self.block_type = block_type
        self.context_dimension = context_dimension
        self.bottleneck_depth = bottleneck_depth
        self.use_attention = use_attention
        self.use_checkpointing = use_checkpointing

        self.time_embedding_dim = self.cfg_mlp.time_embedding_dimension
        self.mlp_hidden_dim = self.cfg_mlp.hidden_dimension
        self.mlp_output_dim = self.cfg_mlp.output_dimension

        self.dropout_chance = 0.0
        
        super().__init__(
            in_channels=self.cfg_unet.in_channels, n_downs=self.cfg_unet.n_downs,
            input_size=config.model.input_size, block_type=self.block_type,
            features=self.cfg_unet.features, init_weights=False, **kwargs)
        
        self.get_embedding_frequency()
        self.init_mlp()
        if self.use_attention: self.define_attention_blocks()
        self.apply(self.init_weights)

    def init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d, nn.GroupNorm)):
            if getattr(m, "weight", None) is not None:
                nn.init.constant_(m.weight, 1.0)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)

    def init_mlp(self) -> None:
        '''Initialized a multilayer perceptron for timestep embedding.'''
        layers = [
            nn.Linear(self.time_embedding_dim, self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_output_dim)]
        self.mlp = nn.Sequential(*layers)

    def get_embedding_frequency(self) -> None:
        '''Precalculates and stores embedding frequency for timesteps.'''
        freq = torch.arange(0, self.time_embedding_dim, 2).float()
        freq /= self.time_embedding_dim
        self.embedding_freq = (1 / (10000 ** freq)).to(self.device)

    def embed_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        '''Embeds timesteps with MLP.
        
        Args:
            timesteps : The timesteps tensor to embed.
        '''
        args = timesteps.unsqueeze(-1) * self.embedding_freq.unsqueeze(0)
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(embeddings.clone())

    def define_attention_blocks(self, num_heads: int=8) -> None:
        '''Initialize self-attention blocks.'''
        self.down_attentions = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(self.channel_map['downs']):
            if i >= len(self.channel_map['downs']) - 1:
                self.down_attentions.append(
                    AttentionBlock(out_ch, num_heads=num_heads))
            else: self.down_attentions.append(nn.Identity())

        self.up_attentions = nn.ModuleList()
        for i in range(1, len(self.channel_map['ups'])):
            if i <= 1:
                out_ch = self.channel_map['ups'][i][1]
                self.up_attentions.append(
                    AttentionBlock(out_ch, num_heads=num_heads))
            else: self.up_attentions.append(nn.Identity())

    def _checkpoint_block(self, block, x, embed_t, context):
        '''Wrapper for checkpointing a block with its arguments.'''
        def custom_forward(x): return block(x, embed_t, context)
        return checkpoint(custom_forward, x, use_reentrant=False)
        
    def _checkpoint_attention(self, attention, x):
        '''Wrapper for checkpointing attention block.'''
        def custom_forward(x): return attention(x)
        return checkpoint(custom_forward, x, use_reentrant=False)

    ##### BLOCK METHOD OVERRIDES #####

    def define_initial_down(self) -> None:
        super().define_initial_down(
            norm='group', activation='silu', residual_activation=None,
            dropout_chance=self.dropout_chance,
            time_embedding_dimension=self.time_embedding_dim,
            context_dimension=self.context_dimension)

    def define_downsampling_blocks(self) -> None:
        '''Defines the layers in the downsampling path.'''
        super().define_downsampling_blocks(
            norm='group', activation='silu', residual_activation=None,
            dropout_chance=self.dropout_chance,
            time_embedding_dimension=self.time_embedding_dim,
            context_dimension=self.context_dimension)

    def define_bottleneck(
            self, activation: str='silu', residual_activation=None,
            norm: str | None='group', stride: int=1, bias: bool=True, 
            use_dropout: bool=False, num_heads: int=8, **kwargs
        ) -> None:
        '''Defines the bottleneck layer.'''
        if self.bottleneck_depth == 1:
            super().define_bottleneck(
                norm=norm, activation=activation, residual_activation=None,
                dropout_chance=self.dropout_chance,
                time_embedding_dimension=self.time_embedding_dim,
                context_dimension=self.context_dimension)
            self.bottleneck = nn.ModuleList([self.bottleneck])
        if self.bottleneck_depth > 1:
            self.bottleneck = nn.ModuleList()
            for i in range(self.bottleneck_depth):
                block = self.block_type(
                    in_channels=self.channel_map['bottleneck'][0], 
                    out_channels=self.channel_map['bottleneck'][1], 
                    upconv_type=self.upconv_type, activation=activation,
                    residual_activation=None, norm=norm, down=True, 
                    stride=stride, bias=bias, use_dropout=use_dropout,
                    dropout_chance=self.dropout_chance, 
                    time_embedding_dimension=self.time_embedding_dim,
                    context_dimension=self.context_dimension)
                self.bottleneck.append(block)

                if self.use_attention and i < self.bottleneck_depth - 1:
                    attn = AttentionBlock(
                        self.channel_map['bottleneck'][1], num_heads=num_heads)
                    self.bottleneck.append(attn)

    def define_upsampling_blocks(self) -> None:
        '''Defines the layers in the upsampling path.'''
        super().define_upsampling_blocks(
            norm='group', activation='silu', residual_activation=None,
            dropout_chance=self.dropout_chance,
            time_embedding_dimension=self.time_embedding_dim,
            context_dimension=self.context_dimension)

    def define_final_up(self) -> None:
        super().define_final_up(
            activation=None, residual_activation=None,
            time_embedding_dimension=self.time_embedding_dim,
            context_dimension=self.context_dimension)

    ##### FORWARD #####

    def _encode(
            self, 
            x: torch.Tensor,
            embed_t: torch.Tensor, 
            context: torch.Tensor
        ) -> torch.Tensor:
        if self.use_checkpointing:
            x = self._checkpoint_block(self.initial_down, x, embed_t, context)
        else: x = self.initial_down(x, embed_t, context)
        self.skips = [x]
        
        for i, down in enumerate(self.downs):
            if self.use_checkpointing:
                x = self._checkpoint_block(down, x, embed_t, context)
            else: x = down(x, embed_t, context)
            if self.use_attention:
                if self.use_checkpointing:
                    x = self._checkpoint_attention(self.down_attentions[i], x)
                else: x = self.down_attentions[i](x)
            self.skips.append(x) 
            
        self.skips.reverse()
        return x

    def _bottleneck(
            self, 
            x: torch.Tensor,
            embed_t: torch.Tensor, 
            context: torch.Tensor
        ) -> torch.Tensor:
        for module in self.bottleneck:
            if isinstance(module, self.block_type):
                if self.use_checkpointing:
                    x = self._checkpoint_block(module, x, embed_t, context)
                else: x = module(x, embed_t, context)
            elif isinstance(module, AttentionBlock):
                if self.use_checkpointing:
                    x = self._checkpoint_attention(module, x)
                else: x = module(x)
        return x
    
    def _decode(
            self, 
            x: torch.Tensor,
            embed_t: torch.Tensor, 
            context: torch.Tensor
        ) -> torch.Tensor:
        for i, up in enumerate(self.ups[1:]):
            skip = self.skips[i]
            x_concat = torch.cat([x, skip], dim=1)
            if self.use_checkpointing:
                x = self._checkpoint_block(up, x_concat, embed_t, context)
            else: x = up(x_concat, embed_t, context)
            if self.use_attention:
                if self.use_checkpointing:
                    x = self._checkpoint_attention(self.up_attentions[i], x)
                else: x = self.up_attentions[i](x)
        return torch.cat([x, self.skips[-1]], dim=1)
        
    def forward(
            self, 
            x: torch.Tensor, 
            timesteps: torch.Tensor,
            context: torch.Tensor | None=None
        ) -> torch.Tensor:
        embed_t = self.embed_timesteps(timesteps)

        x = self._encode(x, embed_t, context)
        x = self._bottleneck(x, embed_t, context)
        x = self._decode(x, embed_t, context)

        if self.use_checkpointing:
            return self._checkpoint_block(self.final_up, x, embed_t, context)
        else: return self.final_up(x, embed_t, context)

