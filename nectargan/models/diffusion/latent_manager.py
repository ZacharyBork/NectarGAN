import json
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field

import torch
from torch.utils.data.dataloader import DataLoader
from diffusers import AutoencoderKL

from nectargan.config import DiffusionConfig
import nectargan.models.diffusion.utils as diffutils
from nectargan.dataset import DiffusionDataset, LatentDataset

@dataclass
class CacheData:
    '''Stores info about shard cache operation at runtime.'''
    shard_count:   int = 0
    current_start: int = 0
    total_length:  int = 0
    output_dir:   Path = None
    manifest:  dict[str, Any] = field(default_factory=dict)
    shard: list[torch.Tensor] = field(default_factory=list)
    
class LatentManager():
    def __init__(
            self,
            config: DiffusionConfig,
            latent_size: int
        ) -> None:
        self.config = config
        self.device = config.common.device
        self.latent_size = latent_size
        self.cache_data = None

        self._init_vae()

    ##### VAE #####

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

    ##### ENCODE / DECODE #####

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

    ##### LATENT_CACHING #####

    def export_shard(self) -> None:
        '''Builds shard file from latent tensors and exports to disk.'''
        if not self.cache_data.shard: return
        
        self.cache_data.shard_count += 1
        t = torch.cat(self.cache_data.shard, dim=0)
        output_path = Path(
            self.cache_data.output_dir, 
            f'shard_{self.cache_data.shard_count}.pt')
        torch.save(t, output_path)

        length = t.shape[0]
        end = self.cache_data.current_start + length
        self.cache_data.manifest['shards'].append({
            'index': self.cache_data.shard_count,
            'filepath': output_path.as_posix(),
            'length': length,
            'start': self.cache_data.current_start,
            'end': end})
        self.cache_data.total_length += length
        self.cache_data.current_start = end
        self.cache_data.shard.clear()

    def _cache_latents(self, batch_size: int, shard_size: int) -> Path:
        '''Loops through Dataloader, encodes tensors to latent space, exports.

        Args:
            batch_size : The batch size to use when caching the latents.
            shard_size : The number of batches to save per shard.

        Returns:
            Path : The path to the cache output directory.
        '''
        dataroot = self.config.dataloader.dataroot
        dataset_path = Path(dataroot, 'train').resolve()
        self.cache_data.output_dir, new = diffutils.init_latent_cache(dataroot)
        if not new: 
            print('Bypassing caching operation...')
            return self.cache_data.output_dir
        
        print(f'Building duplicate Dataloader...\n'
              f'Batch Size : {batch_size}\n'
              f'Shard Size : {shard_size}\n')
        dataloader = DataLoader(
            DiffusionDataset(
                config=self.config, root_dir=dataset_path, is_train=False), 
                batch_size=batch_size, 
                num_workers=self.config.dataloader.num_workers)
        num_batches = len(dataloader)

        self.cache_data.manifest = {
            'dataroot': dataroot,
            'total_length': 0,
            'shard_size': shard_size,
            'shards': []}

        for idx, x in enumerate(dataloader):
            diffutils.print_shard_cache_progress(idx+1, num_batches)
            x = x.to(self.device, non_blocking=True)
            self.cache_data.shard.append(self.encode_to_latent(x).cpu())
            if len(self.cache_data.shard) == shard_size: self.export_shard()
        if not len(self.cache_data.shard) == 0: self.export_shard()
        
        print('Caching complete. Writing manifest...')
        self.cache_data.manifest['total_length'] = self.cache_data.total_length
        with open(Path(self.cache_data.output_dir, 'manifest.json'), 'w') as f:
            f.write(json.dumps(self.cache_data.manifest, indent=4))
        return self.cache_data.output_dir

    def cache_latents(
            self,
            batch_size: int=64,
            shard_size: int=512
        ) -> DataLoader:
        '''Caches latent tensors and builds new dataset to load cache.

        Saves cached latents as shards to new subdirectory of the dataroot from 
        the config used to initialize the LatentManager. The dataloader which
        is returned from this function will mirror the original dataloader
        defined by the config (including using the original batch size, not the
        one used as an input argument for this function).
        
        This means that you can directly overwrite your original dataloader 
        with this functions return, or just use this in lieu of the Trainer's 
        build_dataloader() method.

        Args:
            batch_size : The batch size to use when caching the latents.
            shard_size : The number of batches to save per shard.

        Returns:
            Dataloader : A torch.utils.data.Dataloader initialized to read the
                cached latents.
        '''
        print('Initializing latent precache...')
        self.cache_data = CacheData()
        shard_dir = self._cache_latents(batch_size, shard_size)
        
        print('Building new LatentDataset...')
        latent_dataset = LatentDataset(
            config=self.config, shard_directory=shard_dir)
        
        return DataLoader(
            latent_dataset, batch_size=self.config.dataloader.batch_size, 
            num_workers=self.config.dataloader.num_workers,
            drop_last=True)

