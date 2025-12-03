import time
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nectargan.config import DiffusionConfig
from nectargan.dataset import DiffusionDataset
from nectargan.dataset.streaming_datasets.laion_dataset import LAIONDataset
from nectargan.models import UnetDAE
from nectargan.models.diffusion.data import NoiseParameters

class DiffusionModel(nn.Module):
    def __init__(
            self, 
            config: DiffusionConfig, 
            init_dae: bool=True
        ) -> None:
        super(DiffusionModel, self).__init__()
        self.config = config
        self.device = config.common.device
        self.timesteps = config.model.noise_schedule.timesteps
        
        self.fixed_seed_count = 1
        self.fixed_seeds = []
        self.batch_times = []

        self.noiseparams = NoiseParameters()
        self.noiseparams.build_schedule(
            device=self.device, timesteps=self.timesteps, 
            schedule_type=config.model.noise_schedule.schedule_type,
            cosine_offset=self.config.model.noise_schedule.cosine_offset)

        if init_dae: self._init_autoencoder()

    def _init_dataloader(self) -> None:
        if not self.config.dataloader.streaming.enable:
            dataset = DiffusionDataset(
                config=self.config, 
                root_dir=self.config.dataloader.dataroot, 
                metadata_file=self.config.captions.metadata_file,
                is_train=True, cache_builder=False, recurse=False)
        else:
            streaming_cfg = self.config.dataloader.streaming
            match streaming_cfg.dataset:
                case 'LAION':
                    dataset = LAIONDataset(
                        config=self.config, dataset=streaming_cfg.set,
                        subset=streaming_cfg.subset,
                        split='train', min_aesthetic_score=6.0,
                        max_caption_length=self.config.captions.max_length,
                        max_samples=streaming_cfg.max_samples, 
                        cache_dir=streaming_cfg.cache_directory, 
                        require_login=streaming_cfg.require_login)
                case _: raise ValueError(
                    f'Invalid streaming dataset: {streaming_cfg.dataset}')
                
        self.dataloader = DataLoader(
            dataset, batch_size=self.config.dataloader.batch_size, 
            num_workers=self.config.dataloader.num_workers)

    def _init_autoencoder(self, context_dimension: int | None=None) -> None:
        self.autoencoder = UnetDAE(
            config=self.config, context_dimension=context_dimension
        ).to(self.device, dtype=torch.float32)
        self.opt_dae = optim.Adam(
            self.autoencoder.parameters(), 
            lr=self.config.model.dae.learning_rate.base_rate, 
            betas=self.config.model.dae.betas)
        if self.config.model.mixed_precision:
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
        ) -> tuple[torch.Tensor, torch.Tensor]:
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
            context: torch.Tensor | None=None,
            pred_noise: torch.Tensor | None=None
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
        # Predict noise
        pred_noise = self.autoencoder(x, t, context=context) \
            if pred_noise is None else pred_noise

        # Get parms at timestep (t)
        self.noiseparams(t)
            
        # Sample denoised image x0
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
            else self.config.model.input_size
        shape = (batches, self.config.model.dae.in_channels, size, size)
        with torch.no_grad():
            x = torch.randn(shape).to(self.device) # Generate noise tensor
            for i in reversed(range(self.timesteps)):
                t = torch.full( # Build timesteps for batch
                    (shape[0],), i, device=self.device, dtype=torch.long)
                x = self.p_sample(x, t, idx=i, context=context)
        return x.detach().cpu()
        
    def _trainer_core(
            self, 
            train_step_fn: Callable[[torch.Tensor, torch.Tensor, int], None],
            train_step_kwargs: dict[str, Any]
        ) -> None:
        for idx, x in enumerate(self.dataloader):
            start_time = time.time()
            x: torch.Tensor = x.to(self.device)
            train_step_fn(x, None, idx, **train_step_kwargs)
            batch_time = time.time() - start_time
            self.batch_times.append(batch_time)
