import torch

import nectargan.models.diffusion.utils as diffutils
from nectargan.models import DiffusionModel
from nectargan.models.diffusion.latent_manager import LatentManager

class LatentDiffusionModel(DiffusionModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self._get_latent_spatial_size()
        self._init_latent_manager()
        self.read_from_cache = False

    def _get_latent_spatial_size(self) -> None:
        '''Derive latent spatial size from input size and divisor.'''
        cfg = self.config.model.latent
        if not cfg.override_latent_size:
            size = round(
                self.dae_config.input_size / max(1, cfg.latent_size_divisor))
            diffutils.validate_latent_size(
                size, self.dae_config.input_size, cfg.latent_size_divisor)
        else: size = cfg.latent_size
        self.latent_spatial_size = size
        self.dae_config.input_size = self.latent_spatial_size

    def _init_latent_manager(self) -> None:
        '''Initializes a LatentManager and aliases some of its methods.'''
        self.latent_manager = LatentManager(
            self.config, self.latent_spatial_size)
        self.encode = self.latent_manager.encode_to_latent
        self.decode = self.latent_manager.decode_from_latent
        self.cache_latents = self.latent_manager.cache_latents

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
            idx : Index of the current batch from the dataloader. Only needed
                if `precache_latents` is enabled.

        Returns:
            tuple[torch.Tensor] : The noisy image tensor created by the
                diffusion step, and the noise tensor used for the step.

        Raises:
            ValueError : Is self.precache_latents=True and idx of current batch
                is not provided.
        '''
        with torch.no_grad(): 
            if not self.read_from_cache: 
                x = self.encode(x).to(self.device)
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
        return self.decode(
            super().sample(batches=batches, spatial_size=lss))        

