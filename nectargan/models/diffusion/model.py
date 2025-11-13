import torch
import torch.nn as nn
import torch.optim as optim

from nectargan.models.diffusion.denoising_autoencoder import UnetDAE

class DiffusionModel(nn.Module):
    def __init__(self, train_timesteps: int, device: str) -> None:
        super(DiffusionModel, self).__init__()
        self.device = device
        self.train_timesteps = train_timesteps

        self.dae_learning_rate = 0.0002
        time_embedding_dimension  = 128
        mlp_hidden_dimension = 256
        mlp_output_dimension = 128

        self._build_noise_schedule()

        self.autoencoder = UnetDAE(
            device=self.device,
            time_embedding_dimension=time_embedding_dimension,
            mlp_hidden_dimension=mlp_hidden_dimension,
            mlp_output_dimension=mlp_output_dimension
        ).to(self.device)
        self.opt_dae = optim.Adam(
            self.autoencoder.parameters(), 
            lr=self.dae_learning_rate, betas=(0.5, 0.999))
        self.g_scaler = torch.amp.GradScaler(self.device)

    def _build_noise_schedule(self) -> None:
        self.schedule = {
            'betas': torch.linspace(
                1e-4, 0.02, self.train_timesteps).to(self.device)}
        s = self.schedule
        alphas = s['alphas'] = 1.0 - s['betas']
        s['alphas_cumprod'] = torch.cumprod(alphas, axis=0).to(self.device)

    def q_sample(
            self, 
            x: torch.Tensor, 
            t: torch.Tensor, 
            noise: torch.Tensor = None
        ) -> tuple[torch.Tensor]:
        '''Forward diffusion.
        
        Reference:
            https://arxiv.org/pdf/2006.11239 (eq. 4)
        '''    
        # Generate noise if not provided 
        if noise is None: noise = torch.randn_like(x).to(self.device)

        # Sample noisy image at timestep (t) from input x_0 and noise
        acum = self.schedule['alphas_cumprod']
        x_t = torch.sqrt(acum)[t].view(-1, 1, 1, 1) * x \
            + torch.sqrt(1.0 - acum)[t].view(-1, 1, 1, 1) * noise

        # Return noisy image + noise used (for loss)
        return x_t, noise

    def p_sample(
            self, 
            x: torch.Tensor, 
            t: torch.Tensor, 
            idx: int,
            direct: bool=False
        ) -> torch.Tensor:
        '''Reverse diffusion.

        This sampler has two "modes". If direct=False, it will perform a single 
        reverse diffusion step (see Refs) on x_t to estimate x_(t-1). 

        If direct=True, it will instead estimate the clean image x_0 directly
        from x_t. This is a very aggressive method of prediction, and can cause 
        the model to learn very quickly, but is also very unstable.

        Args:
            x : Noisy image as torch.Tensor
            t : Current timestep as torch.Tensor
            idx : Current index of the denoiser loop.
            direct : See note on sampler modes.
        
        Ref: 
            https://arxiv.org/pdf/2006.11239 (eq. 11)
        '''
        # Predict noise
        dae_output = self.autoencoder(x, t)

        # Get parms at timestep (t)
        alpha_t = self.schedule['alphas'][t].view(-1,1,1,1)
        beta_t = self.schedule['betas'][t].view(-1,1,1,1)
        sqrt_alpha_t = torch.sqrt(alpha_t)

        abar_t = self.schedule['alphas_cumprod'][t].view(-1,1,1,1)
        abar_prev = self.schedule['alphas_cumprod'][
            torch.clamp(t-1, min=0)].view(-1,1,1,1)
        abar_prev = torch.where(
            (t == 0).view(-1,1,1,1), 
            torch.ones_like(abar_prev), abar_prev)
             
        # Sample noisy image x_0
        x_0 = torch.clamp(
            (x - torch.sqrt(1.0 - abar_t) * dae_output) / sqrt_alpha_t, 
            -1.0, 1.0)

        if not direct: # Reverse diffusion, timestep (t) -> (t)-1
            mean = x_0 * (torch.sqrt(abar_prev) * beta_t) / (1.0 - abar_t) \
                 +  x  * (sqrt_alpha_t * (1.0 - abar_prev)) / (1.0 - abar_t)
            var = beta_t * (1.0 - abar_prev) / (1.0 - abar_t)
        else: # Predict clean image directly
            mean, var = x_0, beta_t

        # Return clean image on final step, otherwise noisy image at (t)-1
        if idx == 0: return mean
        else: return mean + torch.sqrt(var) * torch.randn_like(x)

    def sample(
            self, 
            shape: tuple[int]
        ) -> torch.Tensor:
        '''Performs iterative denoising to generate and return an output image.
        
        Args:
            shape : Shape of the image to generate { B, C, W, H }
        '''
        x = torch.randn(shape).to(self.device) # Generate noisy image tensor
        for i in reversed(range(self.train_timesteps)):
            t = torch.full( # Build timesteps for batch
                (shape[0],), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t, idx=i) # Sample reverse diffusion
        return x.detach().cpu()


