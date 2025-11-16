# TO-DO: Fix all hard coded values.

import torch
import torch.nn as nn
import torch.optim as optim

from nectargan.models.diffusion.denoising_autoencoder import UnetDAE
from nectargan.models.diffusion.data import DAEConfig

class DiffusionModel(nn.Module):
    def __init__(
            self, 
            device: str,
            timesteps: int=1000,
            dae_config: DAEConfig=DAEConfig(
                input_size=128, in_channels=3, features=128, n_downs=4, 
                bottleneck_down=True, learning_rate=0.0001)
        ) -> None:
        super(DiffusionModel, self).__init__()
        self.device = device
        self.timesteps = timesteps
        self.dae_config = dae_config

        self._build_noise_schedule()
        self._init_autoencoder()
        
        self.fixed_seed_count = 1
        self.fixed_seeds = []

    def _build_noise_schedule(self) -> None:
        self.schedule = {
            'betas': torch.linspace(
                1e-4, 0.02, self.timesteps).to(self.device)}
        s = self.schedule
        alphas = s['alphas'] = 1.0 - s['betas']
        s['alphas_cumprod'] = torch.cumprod(alphas, axis=0).to(self.device)

    def _init_autoencoder(self) -> None:
        self.autoencoder = UnetDAE(
            device=self.device, dae_config=self.dae_config
        ).to(self.device)
        self.opt_dae = optim.Adam(
            self.autoencoder.parameters(), 
            lr=self.dae_config.learning_rate, betas=self.dae_config.betas)
        self.g_scaler = torch.amp.GradScaler(self.device)

    def _build_fixed_seeds(self, shape: tuple[int]) -> None:
        if len(self.fixed_seeds) != 0: return
        for _ in range(self.fixed_seed_count):
            self.fixed_seeds.append(torch.randn(shape).to(self.device)) 

    def q_sample(
            self, 
            x: torch.Tensor, 
            t: torch.Tensor, 
            noise: torch.Tensor = None
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
            if noise is None: noise = torch.randn_like(x).to(self.device)

            # Sample noisy image at timestep (t) from input x_0 and noise
            acum = self.schedule['alphas_cumprod'][t].view(-1,1,1,1)
            x_t = acum.sqrt() * x + (1.0 - acum).sqrt() * noise

            # Return noisy image + noise used (for loss)
            return x_t, noise
    
    def get_parameters_at_timestep(self, t: torch.Tensor) -> torch.Tensor:
        self.alpha_t = self.schedule['alphas'][t].view(-1,1,1,1)
        self.beta_t = self.schedule['betas'][t].view(-1,1,1,1)
        self.sqrt_alpha_t = torch.sqrt(self.alpha_t)
        
        self.abar_t = self.schedule['alphas_cumprod'][t].view(-1,1,1,1)
        self.inv_abar_t = 1.0 - self.abar_t
        self.sqrt_abar_t = self.abar_t.sqrt()
        self.sqrt_inv_abar_t = torch.sqrt(self.inv_abar_t)
        
        self.abar_prev = self.schedule['alphas_cumprod'][
            torch.clamp(t-1, min=0)].view(-1,1,1,1)
        self.abar_prev = torch.where(
            (t == 0).view(-1,1,1,1), 
            torch.ones_like(self.abar_prev), self.abar_prev)
        self.inv_abar_prev = 1.0 - self.abar_prev
        self.sqrt_abar_prev = torch.sqrt(self.abar_prev)

    def p_sample(
            self, 
            x: torch.Tensor, 
            t: torch.Tensor, 
            idx: int,
            direct: bool=False
        ) -> torch.Tensor:
        '''Reverse diffusion.

        This sampler has two 'modes'. If direct=False, it will perform a single 
        reverse diffusion step (see Refs) on x_t to estimate x_(t-1). 

        If direct=True, it will instead estimate the clean image x_0 directly
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
            dae_output = self.autoencoder(x, t)

            # Get parms at timestep (t)
            self.get_parameters_at_timestep(t)
                
            # Sample noisy image x_0
            x_0 = torch.clamp(
                (x - self.sqrt_inv_abar_t * dae_output) / self.sqrt_abar_t, 
                -4.0, 4.0)

            if not direct: # Reverse diffusion, timestep (t) -> (t)-1
                p1 = (self.sqrt_abar_prev * self.beta_t) / self.inv_abar_t
                p2 = (self.sqrt_alpha_t * self.inv_abar_prev) / self.inv_abar_t
                mean = x_0 * p1 +  x  * p2
                var = self.beta_t * self.inv_abar_prev / self.inv_abar_t
            else: # Predict clean image directly
                mean, var = x_0, self.beta_t

            # Return clean image on final step, otherwise noisy image at (t)-1
            if idx == 0: return mean
            else: return mean + torch.sqrt(var) * torch.randn_like(x)

    def sample(
            self, 
            batches: int=1,
            spatial_size: int | None=None
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
                x = self.p_sample(x, t, idx=i) # Sample reverse diffusion
            return x.detach().cpu()
        

