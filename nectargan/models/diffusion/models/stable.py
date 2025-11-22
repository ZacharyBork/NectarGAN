import torch

from nectargan.models import LatentDiffusionModel
from nectargan.models.diffusion.text_encoder import TextEncoder
from nectargan.config import DiffusionConfig

class StableDiffusionModel(LatentDiffusionModel):
    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__(config=config, init_dae=False)
        self.text_encoder = TextEncoder(
            device=config.common.device,
            max_length=77,
            freeze=True
        ).to(config.common.device)
        self._get_context_dimension()
        self._init_autoencoder()

    def _get_context_dimension(self) -> None:
        context, _ = self.text_encoder(
            ['Those who can imagine anything, can create the impossible.'])
        context_dimension = context.shape[-1]
        self.dae_config.context_dimension = context_dimension


