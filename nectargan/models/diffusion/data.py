from typing import Literal
from dataclasses import dataclass, field

import torch

from nectargan.constants import PI
from nectargan.config import DiffusionConfig

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
                acumprod = abar[1:]
                ones = torch.ones(1, device=device, dtype=acumprod.dtype)
                self.betas = torch.clamp(
                    1 - acumprod / torch.cat([ones, acumprod[:-1]]),
                    1e-8, 0.999)

                self.alphas = 1.0 - self.betas
                self.alphas_cumprod = acumprod

@dataclass
class AverageLossTracker:
    update_freq_visdom:  int
    update_freq_console: int

    steps_visdom:        int = 0
    steps_console:       int = 0
    stored_value_cap:    int = 0

    loss_values: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.stored_value_cap = max(
            self.update_freq_visdom, self.update_freq_console)

    def append_loss_value(self, loss: str, value: float) -> None:
        self.loss_values[loss].insert(0, value)
        self.loss_values[loss] = self.loss_values[loss][:self.stored_value_cap]

    def get_average(
            self, 
            loss: str,
            get_type: Literal['visdom', 'console'],
            precision: int=3
        ) -> float:
        match get_type:
            case 'visdom': divisor = max(1, self.update_freq_visdom)
            case 'console': divisor = max(1, self.update_freq_console)
        average = sum(self.loss_values[loss][:divisor]) / divisor
        return round(average, precision)
