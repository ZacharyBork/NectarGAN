import time
from typing import Any, Callable

import torch

from nectargan.models import LatentDiffusionModel
from nectargan.models.diffusion.text_encoder import TextEncoder
from nectargan.config import DiffusionConfig

class StableDiffusionModel(LatentDiffusionModel):
    def __init__(self, config: DiffusionConfig, init_dae=False) -> None:
        self.model_config = config.model.stable
        super().__init__(config=config, init_dae=init_dae)
        self.text_encoder = TextEncoder(
            device=config.common.device,
            max_length=77,
            freeze=True
        ).to(config.common.device)
        self._get_context_dimension()
        self._init_autoencoder()

    def _init_latent_cache(self) -> None:
        cache_cfg = self.config.model.stable.precache
        if cache_cfg.enable:
            self.train_loader = self.cache_latents(
                batch_size=cache_cfg.batch_size,
                shard_size=cache_cfg.shard_size,
                split='train',
                metadata_file=\
                    '/media/zach/UE/ML/test_data/diffusion/coco2017/'
                    'annotations/captions_train2017_REBUILT.json')
            self.read_from_cache = True

    def _get_context_dimension(self) -> None:
        context, _ = self.text_encoder(
            ['Those who can imagine anything, can create the impossible.'])
        context_dimension = context.shape[-1]
        self.dae_config.context_dimension = context_dimension

    def _trainer_core(
            self, 
            train_step_fn: Callable[[torch.Tensor, torch.Tensor, int], None],
            train_step_kwargs: dict[str, Any]
        ) -> None:
        for idx, (x, y) in enumerate(self.train_loader):
            start_time = time.time()
            image: torch.Tensor = x.to(self.device, dtype=torch.float32)
            context, pooled = self.text_encoder(y)
            context = context.to(self.device, dtype=torch.float32)

            train_step_fn(image, context, idx, **train_step_kwargs)
            batch_time = time.time() - start_time
            self.batch_times.append(batch_time)
