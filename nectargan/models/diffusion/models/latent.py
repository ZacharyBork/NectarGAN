import json
from pathlib import Path

import torch
from torch.utils.data.dataloader import DataLoader
from diffusers import AutoencoderKL

from nectargan.config import Config
from nectargan.models import DiffusionModel
import nectargan.models.diffusion.utils as diffutils
from nectargan.models.diffusion.data import DAEConfig
from nectargan.dataset import UnpairedDataset, LatentDataset

class LatentDiffusionModel(DiffusionModel):
    def __init__(
            self, 
            latent_size_divisor: int=8,
            dae_config: DAEConfig=DAEConfig(
                input_size=256, in_channels=4, features=256, n_downs=2, 
                bottleneck_down=True, learning_rate=0.0001),
            **kwargs
        ) -> None:
        self.dae_config = dae_config
        self._get_latent_spatial_size(latent_size_divisor)
        super().__init__(dae_config=dae_config, **kwargs)
        
        self._init_vae()
        self.read_from_cache = False

    def _get_latent_spatial_size(self, latent_size_divisor: int) -> None:
        '''Derive latent spatial size from input size and divisor.'''
        size = round(self.dae_config.input_size / max(1, latent_size_divisor))
        diffutils.validate_latent_size(
            size, self.dae_config.input_size, latent_size_divisor)
        self.latent_spatial_size = size
        self.dae_config.input_size = self.latent_spatial_size

    def _init_vae(self) -> None:
        '''Initializes a pre-trained VAE from Stability AI.
        
        Ref:
            https://huggingface.co/stabilityai/sd-vae-ft-ema
        '''
        self.vae = AutoencoderKL.from_pretrained(
            'stabilityai/sd-vae-ft-ema', 
            torch_dtype=torch.float16)
        self.vae = self.vae.to(self.device)
        self.vae.eval()
        for p in self.vae.parameters(): p.requires_grad = False
        self.scale = getattr(self.vae.config, 'scaling_factor', 0.18215)        

    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        '''Encodes an input tensor from pixel space to latent space.
        
        Args:
            x : The pixel space tensor to encode.

        Returns:
            torch.Tensor : The resulting tensor encoded to latent space.
        '''
        with torch.no_grad():
            p = next(self.vae.parameters())
            x = x.to(device=p.device, dtype=p.dtype)
            dist = self.vae.encode(x).latent_dist
            return self.scale * dist.mean

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        '''Decodes a tensor from latent space to pixel space.
        
        Args:
            z : The latent space tensor to decode.

        Returns:
            torch.Tensor : The resulting tensor decoded to pixel space.
        '''
        with torch.no_grad():
            p = next(self.vae.parameters())
            z = z.to(device=p.device, dtype=p.dtype)
            return self.vae.decode(z / self.scale).sample.clamp(-1, 1)
        
    def _cache_latents(
            self, 
            config: Config,
            batch_size: int, 
            shard_size: int
        ) -> Path:
        dataroot = config.dataloader.dataroot
        dataset_path = Path(dataroot, 'train').resolve()
        output_dir, is_new = diffutils.init_latent_cache(dataroot)
        if not is_new: 
            print('Bypassing caching operation...')
            return output_dir
        
        print(f'Building duplicate Dataloader...\n'
              f'Batch Size : {batch_size}\n'
              f'Shard Size : {shard_size}\n')
        dataloader = DataLoader(
            UnpairedDataset(
                config=config, root_dir=dataset_path, is_train=True), 
            batch_size=batch_size, num_workers=config.dataloader.num_workers)
        shard: list[torch.Tensor] = []
        shard_count = 0
        current_start = 0
        total_length = 0
        num_batches = len(dataloader)

        manifest = {
            'dataroot': dataroot,
            'total_length': 0,
            'shard_size': shard_size,
            'shards': []}

        def export_shard() -> None:
            nonlocal shard_count, manifest, current_start, total_length
            if not shard: return
            
            shard_count += 1
            t = torch.cat(shard, dim=0)
            output_path = Path(output_dir, f'shard_{shard_count}.pt')
            torch.save(t, output_path)

            length = t.shape[0]
            end = current_start + length
            manifest['shards'].append({
                'index': shard_count,
                'filepath': output_path.as_posix(),
                'length': length,
                'start': current_start,
                'end': end})
            total_length += length
            current_start = end
            shard.clear()

        for idx, x in enumerate(dataloader):
            diffutils.print_shard_cache_progress(idx+1, num_batches)
            x = x.to(self.device, non_blocking=True)
            shard.append(self.encode_to_latent(x).cpu())
            if len(shard) == shard_size: export_shard()
        if not len(shard) == 0: export_shard()
        
        print('Caching complete. Writing manifest...')
        manifest['total_length'] = total_length
        with open(Path(output_dir, 'manifest.json'), 'w') as file:
            file.write(json.dumps(manifest, indent=4))
        return output_dir

    def precache_latents(
            self,
            config: Config,
            batch_size: int=64,
            shard_size: int=512
        ) -> DataLoader:
        print('Initializing latent precache...')
        shard_dir = self._cache_latents(config, batch_size, shard_size)
        
        print('Building new LatentDataset...')
        latent_dataset = LatentDataset(
            config=config, shard_directory=shard_dir)
        
        self.read_from_cache = True
        return DataLoader(
            latent_dataset, batch_size=config.dataloader.batch_size, 
            num_workers=config.dataloader.num_workers,
            drop_last=True)

    def q_sample(
            self, 
            x: torch.Tensor, 
            t: torch.Tensor, 
            noise: torch.Tensor | None=None
        ) -> tuple[torch.Tensor]:
        '''Forward diffusion (see pixel diffusion model q_sample()).
        
        This is just a wrapper for the parent DiffusionModel.q_sample() which
        first encodes the tensor to latent space before performing the forward
        diffusion step.

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
        with torch.no_grad(): 
            if not self.read_from_cache: 
                x = self.encode_to_latent(x).to(self.device)
        return super().q_sample(x, t, noise)
    
    def sample(
            self, 
            batches: int=1, 
            latent_spatial_size: int | None=None
        ) -> torch.Tensor:
        '''Iterative denoising function for latent space tensor.
        
        Args:
            batches : The batch size of the tensor to sample.
            latent_spatial_size : The spatial size of the input tensor for the
                denoising autoencoder, or `None` to use the default latent size
                derived from the dae config input size and the model's latent 
                size devisor.

        Returns:
            torch.Tensor : The final denoised tensor, decoded to pixel space.
        '''
        lss = latent_spatial_size if not latent_spatial_size is None \
            else self.latent_spatial_size
        return self.decode_from_latent(
            super().sample(batches=batches, spatial_size=lss))        

