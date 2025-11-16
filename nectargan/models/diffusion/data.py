from dataclasses import dataclass

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
    

