def validate_latent_size(
        size: int,
        input_size: int,
        latent_size_divisor: int
    ) -> None:
    if size < 2:
        raise ValueError(
            f'Latent size divisor ({latent_size_divisor}) too large for input '
            f'size ({input_size}). The resulting latent tensor would have '
            f'(W=1, H=1).\n\nPlease increase input size, or decrease '
            f'`latent_size_divisor`.')
    elif size < 4:
        print(
            f'Warning: The current input size and latent size divisor will '
            f'result in a very small spacial size for the latent space tensor '
            f'at the DAE\'s bottleneck layer.\n\nThis can lead to very poor '
            f'training results! Please consider increasing input size, or '
            f'decreasing `latent_size_divisor`.')
        
