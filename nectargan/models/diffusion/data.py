from dataclasses import dataclass

from torch import Tensor

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
    

@dataclass
class NoiseParameters:
    alphas:             Tensor | None = None
    betas:              Tensor | None = None
    alphas_cumprod:     Tensor | None = None

    alpha_t:            Tensor | None = None
    beta_t:             Tensor | None = None
    sqrt_alpha_t:       Tensor | None = None

    abar_t:             Tensor | None = None
    sqrt_abar_t:        Tensor | None = None
    inv_abar_t:         Tensor | None = None
    sqrt_inv_abar_t:    Tensor | None = None

    abar_prev:          Tensor | None = None
    sqrt_abar_prev:     Tensor | None = None
    inv_abar_prev:      Tensor | None = None
    sqrt_inv_abar_prev: Tensor | None = None



