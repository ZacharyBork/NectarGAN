import json
from pathlib import Path

import torch
from torch.utils.data.dataloader import DataLoader
from diffusers import AutoencoderKL

from nectargan.config import Config
import nectargan.models.diffusion.utils as diffutils
from nectargan.dataset import UnpairedDataset, LatentDataset


class LatentManager():
    def __init__(
            self,
            config: Config,
            latent_size: int
        ) -> None:
        self.config = config
        self.device = config.common.device
        self.latent_size = latent_size
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

    def _cache_latents(self, batch_size: int, shard_size: int) -> Path:
        dataroot = self.config.dataloader.dataroot
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
                config=self.config, root_dir=dataset_path, is_train=True), 
                batch_size=batch_size, 
                num_workers=self.config.dataloader.num_workers)
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

    def cache_latents(
            self,
            batch_size: int=64,
            shard_size: int=512
        ) -> DataLoader:
        print('Initializing latent precache...')
        shard_dir = self._cache_latents(batch_size, shard_size)
        
        print('Building new LatentDataset...')
        latent_dataset = LatentDataset(
            config=self.config, shard_directory=shard_dir)
        
        return DataLoader(
            latent_dataset, batch_size=self.config.dataloader.batch_size, 
            num_workers=self.config.dataloader.num_workers,
            drop_last=True)

