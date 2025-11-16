import torch
from diffusers import AutoencoderKL

from nectargan.models import DiffusionModel
import nectargan.models.diffusion.utils as diffusion_utils
from nectargan.models.diffusion.data import DAEConfig

class LatentDiffusionModel(DiffusionModel):
    def __init__(
            self, 
            latent_size_divisor: int=8,
            dae_config: DAEConfig=DAEConfig(
                input_size=256, in_channels=4, features=256, n_downs=2, 
                bottleneck_down=True, learning_rate=0.0001),
            **kwargs
        ) -> None:
        self.dae_config = dae_config
        self._get_latent_spatial_size(latent_size_divisor)
        super().__init__(dae_config=dae_config, **kwargs)

        self._init_vae()

    def _get_latent_spatial_size(self, latent_size_divisor: int) -> None:
        '''Derive latent spatial size from input size and divisor.'''
        size = round(self.dae_config.input_size / max(1, latent_size_divisor))
        diffusion_utils.validate_latent_size(
            size, self.dae_config.input_size, latent_size_divisor)
        self.latent_spatial_size = size
        self.dae_config.input_size = self.latent_spatial_size

    def _init_vae(self) -> None:
        '''Initializes a pre-trained VAE from Stability AI.
        
        Ref:
            https://huggingface.co/stabilityai/sd-vae-ft-ema
        '''
        self.vae = AutoencoderKL.from_pretrained(
            'stabilityai/sd-vae-ft-ema', 
            torch_dtype=torch.float32)
        self.vae = self.vae.to(self.device)
        self.vae.eval()
        for p in self.vae.parameters(): p.requires_grad = False
        self.scale = getattr(self.vae.config, 'scaling_factor', 0.18215)        

    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        '''Encodes an input tensor from pixel space to latent space.
        
        Args:
            x : The pixel space tensor to encode.

        Returns:
            torch.Tensor : The resulting tensor encoded to latent space.
        '''
        with torch.no_grad(): 
            x = x.to(self.device)
            dist = self.vae.encode(x).latent_dist
            return self.scale * dist.mean

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        '''Decodes a tensor from latent space to pixel space.
        
        Args:
            z : The latent space tensor to decode.

        Returns:
            torch.Tensor : The resulting tensor decoded to pixel space.
        '''
        with torch.no_grad(): 
            z = z.to(self.device)
            return self.vae.decode(z / self.scale).sample.clamp(-1, 1)
        
    def q_sample(
            self, 
            x: torch.Tensor, 
            t: torch.Tensor, 
            noise: torch.Tensor | None=None
        ) -> tuple[torch.Tensor]:
        '''Forward diffusion (see pixel diffusion model q_sample()).
        
        This is just a wrapper for the parent DiffusionModel.q_sample() which
        first encodes the tensor to latent space before performing the forward
        diffusion step.

        Args:
            x : The current input tensor.
            t : The corresponding timestep tensor.
            noise : The noise tensor to use for the diffusion step, or `None` 
                to generate a random noise tensor.

        Returns:
            tuple[torch.Tensor] : The noisy image tensor created by the
                diffusion step, and the noise tensor used for the step.
        '''
        with torch.no_grad(): x = self.encode_to_latent(x)
        return super().q_sample(x, t, noise)
    
    def sample(
            self, 
            batches: int=1, 
            latent_spatial_size: int | None=None
        ) -> torch.Tensor:
        '''Iterative denoising function for latent space tensor.
        
        Args:
            batches : The batch size of the tensor to sample.
            latent_spatial_size : The spatial size of the input tensor for the
                denoising autoencoder, or `None` to use the default latent size
                derived from the dae config input size and the model's latent 
                size devisor.

        Returns:
            torch.Tensor : The final denoised tensor, decoded to pixel space.
        '''
        lss = latent_spatial_size if not latent_spatial_size is None \
            else self.latent_spatial_size
        return self.decode_from_latent(
            super().sample(batches=batches, spatial_size=lss))        

