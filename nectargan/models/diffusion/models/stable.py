import time
import random
from typing import Any, Callable, Literal

import torch

from nectargan.models import LatentDiffusionModel
from nectargan.models.diffusion.text_encoder import TextEncoder
from nectargan.models.diffusion.blocks import \
    TimeEmbeddedUnetBlock, CrossAttentionUnetBlock
from nectargan.config import DiffusionConfig

class StableDiffusionModel(LatentDiffusionModel):
    def __init__(
            self, 
            config: DiffusionConfig,
            init_dae: bool=True,
            dae_block_type: \
                TimeEmbeddedUnetBlock=CrossAttentionUnetBlock
        ) -> None:
        self.model_config = config.model.stable
        super().__init__(config, False, dae_block_type)
        self.text_encoder = TextEncoder(
            device=config.common.device,
            max_length=self.config.model.captions.max_length,
            freeze=True
        ).to(config.common.device)
        self._get_context_dimension()
        if init_dae: self._init_autoencoder()

    def _init_latent_cache(self) -> None:
        cache_cfg = self.config.model.stable.precache
        if cache_cfg.enable:
            self.train_loader = self.cache_latents(
                batch_size=cache_cfg.batch_size,
                shard_size=cache_cfg.shard_size,
                split=cache_cfg.split,
                metadata_file=self.config.model.stable.metadata_file)
            self.read_from_cache = True

    def _get_context_dimension(self) -> None:
        context, _ = self.text_encoder(
            ['Those who can imagine anything, can create the impossible.'])
        context_dimension = context.shape[-1]
        self.dae_config.context_dimension = context_dimension

    def _drop_captions(
            self, 
            captions: tuple[str], 
            chance: float
        ) -> tuple[str]:
        return tuple([
            caption if random.random() > chance
            else '' for caption in captions])

    def _trainer_core(
            self, 
            train_step_fn: Callable[[torch.Tensor, torch.Tensor, int], None],
            train_step_kwargs: dict[str, Any],
            unconditional_probability: float=0.1
        ) -> None:
        for idx, (x, y) in enumerate(self.train_loader):
            self.captions = y
            
            start_time = time.time()
            image: torch.Tensor = x.to(self.device, dtype=torch.float32)
            captions = self._drop_captions(y, unconditional_probability)

            contexts, _ = self.text_encoder(captions)
            contexts = contexts.to(self.device, dtype=torch.float32)

            train_step_fn(image, contexts, idx, **train_step_kwargs)
            batch_time = time.time() - start_time
            self.batch_times.append(batch_time)

    def _get_contexts(
            self, 
            context: torch.Tensor | None, 
            batches: int
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
        nullcontext = None
        if not context is None: 
            context = context.to(self.device)
            assert context.shape[0] == batches
            nullcontext, _ = self.text_encoder([''] * batches)
            nullcontext = nullcontext.to(self.device)
        else: 
            context, _ = self.text_encoder([''] * batches)
            context = context.to(self.device)
        return context, nullcontext

    def _predict_from_contexts(
            self, 
            x: torch.Tensor,
            t: torch.Tensor,
            context: torch.Tensor,
            nullcontext: torch.Tensor,
            cfg_scale: float
        ) -> torch.Tensor:
        if cfg_scale != 1.0 \
         and context is not None \
         and nullcontext is not None:
            cond = self.autoencoder(x, t, context=context)
            uncond = self.autoencoder(x, t, context=nullcontext)
            pred_noise = uncond + cfg_scale * (cond - uncond)
        else: pred_noise = self.autoencoder(x, t, context=context)
        return pred_noise

    def p_sample(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            idx: int,
            t_prev: torch.Tensor,
            direct: bool=False,
            context: torch.Tensor | None=None,
            nullcontext: torch.Tensor | None=None,
            cfg_scale: float=7.5,
            mode: Literal['DDPM', 'DDIM']='DDIM'
        ) -> torch.Tensor:
        pred = self._predict_from_contexts(
            x, t, context, nullcontext, cfg_scale)
        match mode:
            case 'DDPM': return super().p_sample(x, t, idx, direct, None, pred)
            case 'DDIM':
                self.noiseparams(t)
                x0 = self._predict_x0(x, pred)
                if idx == 0: return x0

                abar = self.noiseparams.alphas_cumprod.to(self.device)
                a_prev = abar[t_prev].to(self.device)
                a_prev = a_prev.view(x.shape[0], 1, 1, 1)
                x_prev = a_prev.sqrt() * x0 + (1.0 - a_prev).sqrt() * pred
                return x_prev
            case _: raise ValueError(f'Invalid sampler mode: {mode}')

    def sample(
            self, 
            batches: int=1,
            latent_spatial_size: int | None=None,
            context: torch.Tensor | None=None,
            cfg_scale: float | None=None,
            inference_steps: int=50,
            mode: Literal['DDPM', 'DDIM']='DDIM'
        ) -> torch.Tensor:
        if not mode == 'DDIM': inference_steps = self.timesteps
        lss = latent_spatial_size or self.latent_manager.latent_size
        cfg_scale = cfg_scale or self.model_config.cfg_scale
        shape = (batches, self.dae_config.in_channels, lss, lss)
        
        self._build_fixed_seeds(shape)
        with torch.no_grad():
            x = torch.randn(shape, device=self.device)
            context, nullcontext = self._get_contexts(context, batches)

            steps = torch.linspace(
                self.timesteps - 1, 0, inference_steps,
                dtype=torch.long, device=self.device)
            steps = steps.tolist()

            for i, t_idx in enumerate(steps):
                idx = len(steps) - 1 - i
                t = torch.full(
                    (batches,), t_idx, device=self.device, dtype=torch.long)
                t_prev = torch.full(
                    (batches,), steps[min(len(steps)-1, i+1)], 
                    device=self.device, dtype=torch.long)
                x = self.p_sample(
                    x, t, idx=idx, t_prev=t_prev, context=context,
                    nullcontext=nullcontext, cfg_scale=cfg_scale)
        
        return self.decode(x.detach().cpu())
