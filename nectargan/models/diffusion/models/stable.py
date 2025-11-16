from nectargan.models import LatentDiffusionModel
from nectargan.models.diffusion.data import DAEConfig


class StableDiffusionModel(LatentDiffusionModel):
    def __init__(
            self, 
            latent_size_divisor: int=8, 
            dae_config: DAEConfig=DAEConfig(
                input_size=128, in_channels=4, features=128, n_downs=1, 
                bottleneck_down=True, learning_rate=0.0001),
            **kwargs
        ) -> None:
        super().__init__(latent_size_divisor, dae_config, **kwargs)







