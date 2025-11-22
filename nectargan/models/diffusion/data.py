from typing import Literal
from dataclasses import dataclass

import torch

from nectargan.constants import PI

@dataclass
class DAEConfig:
    input_size:           int = 128
    in_channels:          int = 3
    features:             int = 96
    n_downs:              int = 5
    bottleneck_down:     bool = True
    learning_rate:      float = 0.0001
    betas:       tuple[float] = (0.9, 0.999)
    time_embed_dimension: int = 128
    mlp_hidden_dimension: int = 256
    mlp_output_dimension: int = 128
    context_dimension:    int = None
    
@dataclass
class NoiseParameters:
    alphas:             torch.Tensor | None = None
    betas:              torch.Tensor | None = None
    alphas_cumprod:     torch.Tensor | None = None

    alpha_t:            torch.Tensor | None = None
    beta_t:             torch.Tensor | None = None
    sqrt_alpha_t:       torch.Tensor | None = None

    abar_t:             torch.Tensor | None = None
    sqrt_abar_t:        torch.Tensor | None = None
    inv_abar_t:         torch.Tensor | None = None
    sqrt_inv_abar_t:    torch.Tensor | None = None

    abar_prev:          torch.Tensor | None = None
    sqrt_abar_prev:     torch.Tensor | None = None
    inv_abar_prev:      torch.Tensor | None = None
    sqrt_inv_abar_prev: torch.Tensor | None = None

    def __call__(self, *args, **kwds) -> None:
        '''Updates noise parameters from input timestep tensor.'''
        t = args[0]
        self.alpha_t = self.alphas[t].view(-1,1,1,1)
        self.beta_t = self.betas[t].view(-1,1,1,1)
        self.sqrt_alpha_t = torch.sqrt(self.alpha_t)
        
        self.abar_t = self.alphas_cumprod[t].view(-1,1,1,1)
        self.inv_abar_t = 1.0 - self.abar_t
        self.sqrt_abar_t = self.abar_t.sqrt()
        self.sqrt_inv_abar_t = torch.sqrt(self.inv_abar_t)
        
        self.abar_prev = self.alphas_cumprod[
            torch.clamp(t-1, min=0)].view(-1,1,1,1)
        self.abar_prev = torch.where(
            (t == 0).view(-1,1,1,1), torch.ones_like(
                self.abar_prev), self.abar_prev)
        self.inv_abar_prev = 1.0 - self.abar_prev
        self.sqrt_abar_prev = torch.sqrt(self.abar_prev)

    def build_schedule(
            self, 
            device: str,
            timesteps: torch.Tensor,
            schedule_type: Literal['linear', 'cosine'],
            cosine_offset: float=0.008
        ) -> None:
        match schedule_type:
            case 'linear':
                self.betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
                self.alphas = 1.0 - self.betas
                self.alphas_cumprod = torch.cumprod(
                    self.alphas, axis=0).to(device)
            case 'cosine':
                steps = timesteps + 1
                offset = cosine_offset
                x = torch.linspace(0, timesteps, steps, device=device)
                abar = torch.pow(torch.cos(
                    ((x / timesteps + offset) / (1 + offset)) * PI/2), 2)
                abar = abar / abar[0]

                self.betas = torch.clamp(
                    1.0 - (abar[1:] / abar[:-1]), 1e-8, 0.999)
            
                self.alphas = 1.0 - self.betas
                self.alphas_cumprod = torch.cumprod(
                    self.alphas, dim=0).to(device)



