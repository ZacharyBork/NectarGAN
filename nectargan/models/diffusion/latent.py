import time
import random
from typing import Literal, Callable, Any

import torch
from torch.utils.data import DataLoader

from nectargan.models import PixelDiffusionModel
from nectargan.models.diffusion.text_encoder import TextEncoder
from nectargan.models.unet.blocks import \
    TimeEmbeddedUnetBlock, CrossAttentionUnetBlock
from nectargan.latent.latent_manager import VAE
from nectargan.config import DiffusionConfig

from nectargan.latent import latent_utils
from nectargan.dataset import ImageTextDataset
from dataset.cache_loading.loaders import CacheLoader, CacheShardHotloader

class LatentDiffusionModel(PixelDiffusionModel):
    def __init__(
            self, 
            config: DiffusionConfig,
            testing: bool = False
        ) -> None:
        '''Initialized a LatentDiffusionModel.
        
        Args:
            config : The DiffusionConfig to use for the model.
            testing : Enables inference-only testing mode. Disables automatic
                dataloader initialization.
        '''
        super().__init__(config, init_unet=False, testing=testing)
        self.use_captions = self.config.captions.use_captions
        self.read_from_cache = self.config.latents.cache.read_from_cache
        self.latent_size = latent_utils.get_latent_spatial_size(config)
        self.dtype = torch.float16 if self.config.model.mixed_precision \
            else torch.float32

        self._init_vae()
        self._init_unet()
        if not self.testing: self._init_dataloader()
        
    ##### INIT #####

    def _init_text_encoder(self) -> None:
        device = self.config.common.device
        C = self.config.captions
        self.text_encoder = TextEncoder(
            device=device, model_name=C.encoder_model, 
            max_length=C.max_length, freeze=C.freeze_encoder)
        self.text_encoder = self.text_encoder.to(device)

    def _init_unet(self) -> None:
        if self.use_captions:
            block_type = CrossAttentionUnetBlock
            self._init_text_encoder()
            context, _ = self.text_encoder(
                ['Those who can imagine anything, can create the impossible.'])
            context_dimension=context.shape[-1]
        else:
            block_type = TimeEmbeddedUnetBlock
            context_dimension = None
        super()._init_unet(
            block_type=block_type, context_dimension=context_dimension)
        
        parameters = sum(p.numel() for p in self.unet.parameters())
        print(f'Parameter Count: {parameters}')

    def _init_vae(self) -> None:
        cfg_l = self.config.latents
        self.vae = VAE(
            device=cfg_l.vae_device, dtype=cfg_l.vae_dtype, 
            model=cfg_l.vae, scaling_factor=cfg_l.scaling_factor)
        self.encode = self.vae.encode_to_latent
        self.decode = self.vae.decode_from_latent
        
    def _init_dataloader(self) -> None:
        '''Initializes a dataloader for the model.
        
        If latent pre-caching is enabled in the DiffusionConfig, this method
        will also run the pre-caching pass.
        '''
        if not self.read_from_cache: super()._init_dataloader()
        else:
            loader = self.config.dataloader
            cache_cfg = self.config.latents.cache
            if self.config.captions.use_captions:
                dataset = ImageTextDataset(
                    shard_directory=cache_cfg.cache_directory,
                    metadata_file=self.config.captions.metadata_file,
                    latent_size=self.latent_size)
            else:
                match cache_cfg.loader_type:
                    case 'CacheLoader':
                        dataset = CacheLoader(
                            shard_directory=cache_cfg.cache_directory, 
                            load_size=self.latent_size,
                            crop_type=loader.crop_type)
                    case 'CacheShardHotloader':
                        dataset = CacheShardHotloader(
                            shard_directory=cache_cfg.cache_directory, 
                            load_size=self.latent_size,
                            crop_type=loader.crop_type)
                    case _:
                        raise ValueError(
                            f'Invalid cache loader type: '
                            f'{cache_cfg.loader_type}')
            
            self.dataloader = DataLoader(
                dataset, batch_size=loader.batch_size,
                num_workers=loader.num_workers, drop_last=loader.drop_last, 
                pin_memory=loader.pin_memory, shuffle=loader.shuffle)

    ##### CAPTIONS & CONTEXTS #####

    def _drop_captions(
            self, 
            captions: tuple[str], 
            chance: float
        ) -> tuple[str]:
        '''Randomly drops captions from a batch.

        Dropping captions is achieved by just setting the given caption to an
        empty string.

        Args:
            captions : The captions for the batch.
            chance : The chance of dropping any given caption (0:1).
        
        Returns:
            tuple[str] : The updated list of captions.
        '''
        return tuple([
            caption if random.random() > chance
            else '' for caption in captions])

    def _get_nullcontexts(
            self, 
            context: torch.Tensor | None, 
            batches: int
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
        '''Builds nullcontexts for batch.

        Args:
            context : If None, this method will build a nullcontext for every
                image in the batch. If a torch.Tensor is passed for this
                argument, however, it will instead build a batch of null
                contexts for the contexts.
            batches : The batch size for the null contexts tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None] : If contexts is not None,
                this method returns a tuple containing the input contexts, cast
                to the current device, and the corresponding nullcontexts. If
                contexts is None, it will return a tuple containing a null
                contexts tensor with batch size=batches, and a Nonetype object
                for the second index.
        '''
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
            context: torch.Tensor | None,
            nullcontext: torch.Tensor | None,
            cfg_scale: float | None
        ) -> torch.Tensor:
        '''Runs UNet to predict noise with context.

        Args:
            x : The input image tensor.
            y : The corresponding timesteps tensor.
            context : The contexts tensor.
            nullcontext : The corresponding nullcontext tensor.
            cfg_scale : The classifier free guidance scale to use for
                denoising.
        
        Returns:
            torch.Tensor : The predicted noise tensor.
        '''
        if cfg_scale != 1.0 \
         and not context is None \
         and not nullcontext is None:
            cond = self.unet(x, t, context=context)
            uncond = self.unet(x, t, context=nullcontext)
            pred_noise = uncond + cfg_scale * (cond - uncond)
            del cond, uncond
        else: pred_noise = self.unet(x, t, context=context)
        return pred_noise
    
    ##### SAMPLING #####

    def sample_ddim(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            t_prev: torch.Tensor,
            idx: int,
            predictions: torch.Tensor | None=None
        ) -> torch.Tensor:
        self.noiseparams(t)
        x0 = self._predict_x0(x, predictions)
        if idx == 0: return x0

        abar = self.noiseparams.alphas_cumprod
        a_prev = abar[t_prev].view(x.shape[0], 1, 1, 1)

        if self.config.model.sampling.ddim_recompute_epsilon:
            a_t = abar[t].view(x.shape[0], 1, 1, 1)
            direction = (x - a_t.sqrt() * x0) / (1.0 - a_t).sqrt()   
            x_prev = a_prev.sqrt() * x0 \
                + (1.0 - a_prev).sqrt() * direction
        else: x_prev = a_prev.sqrt() * x0 + (1.0 - a_prev).sqrt() * predictions
        return x_prev

    def q_sample(
            self, 
            x: torch.Tensor, 
            t: torch.Tensor, 
            noise: torch.Tensor | None=None
        ) -> tuple[torch.Tensor, torch.Tensor]:
        '''Forward diffusion (see pixel diffusion model q_sample()).
        
        This is just a wrapper for the parent PixelDiffusionModel.q_sample() 
        which first encodes the tensor to latent space before performing the 
        forward diffusion step.

        Args:
            x : The current input tensor.
            t : The corresponding timestep tensor.
            noise : The noise tensor to use for the diffusion step, or `None` 
                to generate a random noise tensor.
            idx : Index of the current batch from the dataloader. Only needed
                if `precache_latents` is enabled.

        Returns:
            tuple[torch.Tensor] : The noisy image tensor created by the
                diffusion step, and the noise tensor used for the step.

        Raises:
            ValueError : Is self.precache_latents=True and idx of current batch
                is not provided.
        '''
        if not self.read_from_cache: 
            with torch.no_grad(): x = self.encode(x).to(self.device)
        return super().q_sample(x, t, noise)
    
    def p_sample(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            t_prev: torch.Tensor,
            idx: int,
            direct: bool=False,
            predictions: torch.Tensor | None=None,
            mode: Literal['DDPM', 'DDIM']='DDIM'
        ) -> torch.Tensor:
        '''Reverse diffusion with DDPM or DDIM prediction.
        
        Args:
            x : Noisy image as torch.Tensor
            t : Current timestep as torch.Tensor
            t_prev : The previous timestep as torch.Tensor.
            idx : Current index of the denoiser loop.
            direct : See PixelDiffusionModel.p_sample(). Only applies to DDPM.
            predictions : The predicted noise tensors from the UNet.
            mode : What sampling mode to use ["DDPM", "DDIM"].
            
        Returns:
            torch.Tensor : The denoised image tensor.
        '''
        P = predictions
        match mode:
            case 'DDPM': return super().p_sample(x, t, idx, direct, None, P)
            case 'DDIM': return self.sample_ddim(x, t, t_prev, idx, P)
            case _: raise ValueError(f'Invalid sampler mode: {mode}')
    
    def sample(
            self, 
            batches: int=1,
            latent_spatial_size: int | None=None,
            context: torch.Tensor | None=None,
            cfg_scale: float=7.5,
            inference_steps: int=100,
            mode: Literal['DDPM', 'DDIM']='DDIM'
        ) -> torch.Tensor:
        '''Iterative denoising function.
        
        Args:
            batches : The batch size of the tensor to sample.
            latent_spatial_size : The spatial size of the latent-space input 
                tensor for the denoising autoencoder, or `None` to use the 
                "latent_size" from the config file.
            context : The contexts tensor for the current batch, or None to use
                null contexts.
            cfg_scale : The classifier free guidance scale to use for
                denoising.
            inference_steps : How many steps to use for denoising. Only used if
                mode=DDPM, otherwise the timesteps from the noise_schedule 
                section of the config will be used.
            mode : What sampling mode to use ["DDPM", "DDIM"].

        Returns:
            torch.Tensor : The denoised image tensor.
        '''
        if not mode == 'DDIM': inference_steps = self.timesteps
        lss = latent_spatial_size or self.latent_size
        shape = (batches, self.config.model.unet.in_channels, lss, lss)
        
        with torch.inference_mode():
            x = torch.randn(shape, device=self.device)
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
                
                if self.use_captions:
                    context, nullcontext = self._get_nullcontexts(
                        context, batches)
                    predictions = self._predict_from_contexts(
                        x, t, context, nullcontext, cfg_scale)
                else: predictions = self.unet(x, t, context=None)
                
                x = self.p_sample(
                    x, t, idx=idx, t_prev=t_prev, 
                    predictions=predictions, mode=mode)
                
                del predictions, t, t_prev
                
        z = self.decode(x)
        result = torch.clamp((z + 1) * 0.5, 0.0, 1.0).detach().cpu()
        del z, x
        if self.device == 'cuda': torch.cuda.empty_cache()
        return result
    
    ##### TRAINER CORE #####

    def trainer_core(
            self, 
            train_step_fn: Callable[[torch.Tensor, torch.Tensor, int], None],
            train_step_kwargs: dict[str, Any] | None=None,
            unconditional_probability: float=0.1
        ) -> None:
        '''Trainer core callback for conditional diffusion.
        
        Args:
            train_step_fn : Train step function, run once per batch. Passed by
                "Trainer.train()".
            train_step_kwargs : Optional keyword args for train step function.
        '''
        if not self.use_captions:
            return super().trainer_core(train_step_fn, train_step_kwargs)
        for idx, data in enumerate(self.dataloader):
            start_time = time.time()
            if self.config.captions.use_captions:
                image: torch.Tensor = data[0]
                image = image.to(
                    self.device, dtype=self.dtype, non_blocking=True)
                self.captions = data[1]
                captions = self._drop_captions(
                    self.captions, unconditional_probability)
                contexts, _ = self.text_encoder(captions)
                contexts = contexts.to(self.device, dtype=self.dtype)
            else: 
                image = data
                contexts = None

            train_step_fn(image, contexts, idx, **train_step_kwargs)
            batch_time = time.time() - start_time
            self.batch_times.append(batch_time) 

