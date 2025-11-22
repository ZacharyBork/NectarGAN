import torch

from nectargan.models import DiffusionModel
from nectargan.dataset.utility import LatentManager
from nectargan.config import DiffusionConfig

class LatentDiffusionModel(DiffusionModel):
    def __init__(self, config: DiffusionConfig, init_dae=False) -> None:
        if self.model_config is None: self.model_config = config.model.latent
        super().__init__(config=config, init_dae=False)
        
        self._init_latent_manager()
        self.read_from_cache = False
        self._init_latent_cache()

    def _init_latent_manager(self) -> None:
        '''Initializes a LatentManager and aliases some of its methods.'''
        self.latent_manager = LatentManager(self.config)
        self.encode = self.latent_manager.encode_to_latent
        self.decode = self.latent_manager.decode_from_latent
        self.cache_latents = self.latent_manager.cache_latents

    def _init_latent_cache(self) -> None:
        cache_cfg = self.config.model.stable.precache
        if cache_cfg.enable:
            if cache_cfg.enable:
                self.train_loader = self.cache_latents(
                    batch_size=cache_cfg.batch_size,
                    shard_size=cache_cfg.shard_size,
                    split='train')
                self.read_from_cache = True

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
            latent_spatial_size: int | None=None,
            **kwargs
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
            else self.latent_manager.latent_size
        return self.decode(
            super().sample(batches=batches, spatial_size=lss, **kwargs))        

