import torch
import torch.nn as nn
import torch.optim as optim
from typing import Literal

from nectargan.constants import PI
from nectargan.config import DiffusionConfig
from nectargan.models import UnetDAE
from nectargan.models.diffusion.data import DAEConfig, NoiseParameters

class DiffusionModel(nn.Module):
    def __init__(
            self, 
            config: DiffusionConfig, 
            init_dae: bool=True,
            mixed_precision: bool=True
        ) -> None:
        super(DiffusionModel, self).__init__()
        self.config = config
        self.device = config.common.device
        self.model_type = config.model.model_type
        self.timesteps = config.model.common.timesteps
        self.mixed_precision = mixed_precision
        self.dae_config = self._init_dae_config()
        self.noiseparams = NoiseParameters()

        self._build_noise_schedule(
            schedule_type=config.model.common.noise_schedule)
        if init_dae: self._init_autoencoder()
        
        self.fixed_seed_count = 1
        self.fixed_seeds = []

    def _init_dae_config(self) -> DAEConfig:
        common_cfg = self.config.model.common
        match self.model_type:
            case 'pixel': model_cfg = self.config.model.pixel
            case 'latent': model_cfg = self.config.model.latent
            case 'stable': model_cfg = self.config.model.stable
            case _: raise ValueError(f'Invalid model_type: {self.model_type}')
        return DAEConfig(
            input_size=model_cfg.input_size,
            in_channels=model_cfg.dae.in_channels,
            features=model_cfg.dae.features,
            n_downs=model_cfg.dae.n_downs,
            learning_rate=common_cfg.dae.learning_rate.initial,
            betas=(common_cfg.dae.betas[0], common_cfg.dae.betas[1]),
            time_embed_dimension=common_cfg.dae.time_embedding_dimension,
            mlp_hidden_dimension=common_cfg.dae.mlp_hidden_dimension,
            mlp_output_dimension=common_cfg.dae.time_embedding_dimension)

    def _build_noise_schedule(
            self, 
            schedule_type: Literal['linear', 'cosine']
        ) -> None:
        n = self.noiseparams
        match schedule_type:
            case 'linear':
                n.betas = torch.linspace(
                    1e-4, 0.02, self.timesteps).to(self.device)
                n.alphas = 1.0 - n.betas
                n.alphas_cumprod = torch.cumprod(
                    n.alphas, axis=0).to(self.device)
            case 'cosine':
                steps = self.timesteps + 1
                offset = self.config.model.common.cosine_offset
                x = torch.linspace(
                    0, self.timesteps, steps, device=self.device)
                abar = torch.pow(torch.cos(
                    ((x / self.timesteps + offset) / (1 + offset)) * PI/2), 2)
                abar = abar / abar[0]

                n.betas = torch.clamp(
                    1.0 - (abar[1:] / abar[:-1]), 1e-8, 0.999)
            
                n.alphas = 1.0 - n.betas
                n.alphas_cumprod = torch.cumprod(
                    n.alphas, dim=0).to(self.device)

    def _init_autoencoder(self) -> None:
        self.autoencoder = UnetDAE(
            device=self.device, dae_config=self.dae_config
        ).to(self.device, dtype=torch.float32)
        self.opt_dae = optim.Adam(
            self.autoencoder.parameters(), 
            lr=self.dae_config.learning_rate, betas=self.dae_config.betas)
        if self.mixed_precision:
            self.g_scaler = torch.amp.GradScaler(self.device)

    def _build_fixed_seeds(self, shape: tuple[int]) -> None:
        if len(self.fixed_seeds) != 0: return
        for _ in range(self.fixed_seed_count):
            self.fixed_seeds.append(torch.randn(shape).to(self.device)) 

    def q_sample(
            self, 
            x: torch.Tensor, 
            t: torch.Tensor, 
            noise: torch.Tensor=None
        ) -> tuple[torch.Tensor]:
        '''Forward diffusion.
        
        Args:
            x : The current input tensor.
            t : The corresponding timestep tensor.
            noise : The noise tensor to use for the diffusion step, or `None` 
                to generate a random noise tensor.

        Returns:
            tuple[torch.Tensor] : The noisy image tensor created by the
                diffusion step, and the noise tensor used for the step.

        Ref:
            https://arxiv.org/pdf/2006.11239 (2)
        '''    
        with torch.no_grad():
            # Generate noise if not provided 
            if noise is None: noise = torch.randn_like(x)

            # Sample noisy image at timestep (t) from input x0 and noise
            acum = self.noiseparams.alphas_cumprod[t].view(-1,1,1,1)
            x_t = acum.sqrt() * x + (1.0 - acum).sqrt() * noise

            # Return noisy image + noise used (for loss)
            return x_t, noise

    def _predict_x0(
            self, 
            x: torch.Tensor,
            pred_noise: torch.Tensor, 
            range: float=4.0
        ) -> torch.Tensor:
        n = self.noiseparams
        return torch.clamp(
            (x - n.sqrt_inv_abar_t * pred_noise) / n.sqrt_abar_t, 
            -range, range)
    
    def p_sample(
            self, 
            x: torch.Tensor, 
            t: torch.Tensor, 
            idx: int,
            direct: bool=False,
            context: torch.Tensor | None=None
        ) -> torch.Tensor:
        '''Reverse diffusion.

        This sampler has two 'modes'. If direct=False, it will perform a single 
        reverse diffusion step (see Refs) on x_t to estimate x_(t-1). 

        If direct=True, it will instead estimate the clean image x0 directly
        from x_t. This is a very aggressive method of prediction, and can cause 
        the model to learn very quickly, but is also very unstable.

        Args:
            x : Noisy image as torch.Tensor
            t : Current timestep as torch.Tensor
            idx : Current index of the denoiser loop.
            direct : See note on sampler modes.

        Returns:
            torch.Tensor : The resulting denoised image tensor from the reverse
                diffusion step.
        
        Ref: 
            https://arxiv.org/pdf/2006.11239 (3.2)
        '''
        with torch.no_grad():
            # Predict noise
            pred_noise = self.autoencoder(x, t, context=context)

            # Get parms at timestep (t)
            self.noiseparams(t)
                
            # Sample noisy image x0
            x0 = self._predict_x0(x, pred_noise)

            n = self.noiseparams
            if not direct: # Reverse diffusion, timestep (t) -> (t)-1
                p1 = (n.sqrt_abar_prev * n.beta_t) / n.inv_abar_t
                p2 = (n.sqrt_alpha_t * n.inv_abar_prev) / n.inv_abar_t
                mean = x0 * p1 +  x  * p2
                var = n.beta_t * n.inv_abar_prev / n.inv_abar_t
            else: # Predict clean image directly
                mean, var = x0, n.beta_t

            # Return clean image on final step, otherwise noisy image at (t)-1
            if idx == 0: return mean
            else: return mean + torch.sqrt(var) * torch.randn_like(x)

    def sample(
            self, 
            batches: int=1,
            spatial_size: int | None=None,
            context: torch.Tensor | None=None
        ) -> torch.Tensor:
        '''Performs iterative denoising to generate and return an output image.
        
        Args:
            batches : The batch size of the tensor to sample.
            spatial_size : The spatial size of the input tensor for the
                denoising autoencoder, or `None` to use the input size from the
                DAE config.

        Returns:
            torch.Tensor : The final denoised tensor, decoded to pixel space.
        '''
        size = spatial_size if not spatial_size is None \
            else self.dae_config.input_size
        shape = (batches, self.dae_config.in_channels, size, size)
        self._build_fixed_seeds(shape)
        with torch.no_grad():
            x = torch.randn(shape).to(self.device) # Generate noise tensor
            for i in reversed(range(self.timesteps)):
                t = torch.full( # Build timesteps for batch
                    (shape[0],), i, device=self.device, dtype=torch.long)
                x = self.p_sample(x, t, idx=i, context=context)
            return x.detach().cpu()
        

