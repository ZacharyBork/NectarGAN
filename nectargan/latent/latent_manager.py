import sys
import json
from os import PathLike
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field
from contextlib import nullcontext

import torch
from torch.utils.data.dataloader import DataLoader
from diffusers import AutoencoderKL

from nectargan.latent import latent_utils
from nectargan.config import DiffusionConfig 
from nectargan.dataset import DiffusionDataset, ImageTextDataset
from dataset.cache_loading.loaders import CacheLoader, CacheShardHotloader

@dataclass
class CacheData:
    '''Stores info about shard cache operation at runtime.'''
    shard_count:   int = 0
    current_start: int = 0
    total_length:  int = 0
    output_dir:   Path = None
    manifest:  dict[str, Any] = field(default_factory=dict)
    shard: list[torch.Tensor] = field(default_factory=list)
    file_names:     list[str] = field(default_factory=list)

class VAE():
    def __init__(
            self,
            device: str = 'cuda',
            dtype: str = 'float16',
            model: str = 'stabilityai/sd-vae-ft-ema',
            scaling_factor: float = 0.18215,
            decoder_range: tuple[float, float] = (-1.0, 1.0),
            require_grad: bool = False
        ) -> None:
        self.device = device
        self.model = model
        self.scaling_factor = scaling_factor
        self.decoder_range = decoder_range
        self.require_grad = require_grad

        match dtype:
            case 'float16': self.dtype = torch.float16
            case 'float32': self.dtype = torch.float32
            case _: raise ValueError(f'VAE recieved unexpected dtype: {dtype}')

        self._init_vae()

    def _gradient_context(self) -> None:
        return nullcontext() if self.require_grad else torch.no_grad()

    def _init_vae(self) -> None:
        '''Initializes a pre-trained VAE from Stability AI.
        
        Ref:
            https://huggingface.co/stabilityai/sd-vae-ft-ema
        '''
        vae = AutoencoderKL.from_pretrained(self.model, torch_dtype=self.dtype)
        self.vae = vae.to(self.device)
        self.vae.eval()
        for p in self.vae.parameters(): p.requires_grad = False
        self.scale = getattr(
            self.vae.config, 'scaling_factor', self.scaling_factor)        

    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        '''Encodes an input tensor from pixel space to latent space.
        
        Args:
            x : The pixel space tensor to encode.

        Returns:
            torch.Tensor : The resulting tensor encoded to latent space.
        '''
        with self._gradient_context():
            x = x.to(device=self.device, dtype=self.dtype)
            dist = self.vae.encode(x).latent_dist
            return self.scale * dist.sample()

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        '''Decodes a tensor from latent space to pixel space.
        
        Args:
            z : The latent space tensor to decode.

        Returns:
            torch.Tensor : The resulting tensor decoded to pixel space.
        '''
        with self._gradient_context():
            z = z.to(device=self.device, dtype=self.dtype)
            z = self.vae.decode(z / self.scale).sample
            return z.clamp(self.decoder_range[0], self.decoder_range[1])


class LatentManager():
    def __init__(
            self,
            dataroot: PathLike,
            device: str = 'cuda',
            dtype: str = 'float16',
            model: str = 'stabilityai/sd-vae-ft-ema',
            latent_spatial_size: int = 64,
            batch_size: int = 32,
            shard_size: int = 4096,
            num_workers: int = 0,
            store_file_names: bool = True
        ) -> None:
        self.dataroot = Path(dataroot).resolve()
        self.device = device
        self.latent_size = latent_spatial_size
        self.input_size = 8 * self.latent_size
        self.batch_size = batch_size
        self.shard_size = shard_size
        self.num_workers = num_workers
        self.store_file_names = store_file_names

        self.cache_data: CacheData = None
        self.vae = VAE(device=device, dtype=dtype, model=model)

    ##### LATENT_CACHING #####

    def _validate_metadata_file(self, metadata_file: PathLike) -> Path:
        '''Converts metadata_file to pathlib.Path and verifies existence.
        
        Args:
            metadata_file : A PathLike object pointing to the metadata file for
                the dataset being cached.

        Returns:
            Path : The validated metadata file as a pathlib.Path.
        '''
        metadata_file = Path(metadata_file)
        if not metadata_file.exists():
            raise FileNotFoundError(
                f'Unable to locate metadata file at path: '
                f'{metadata_file.as_posix()}')
        return metadata_file

    def _export_shard(self, stack: bool=True) -> None:
        '''Builds shard file from latent tensors and exports to disk.

        Note: This method has two methods of operation, based on the value of
        `stack`. If `stack` is True, the tensors will be stacked on dim 0 
        before being saved to disk. If False, the list of latent tensors will 
        be saved directly. This argument is set automatically by the 
        `_iterate_dataloader()` method to True for any batch size > 1, 
        otherwise False. 
        
        This allows you to buiild caches from datasets which do not have a
        standardizes resolution for their images by setting batch size for the
        caching operation to 1. This is slower to cache, but it does not 
        require a pre-crop which means the latents can instead be cropped 
        randomly during training and all data is preserved.

        If your dataset images are all of the same resolution though, you can
        instead use a much larger batch size to cache the dataset more quickly.

        Args:
            stack : If True, the tensors will be stacked on dim 0 before being
                saved to disk. See Note for more info. 
        '''
        if not self.cache_data.shard: return
        self.cache_data.shard_count += 1

        if stack: 
            t = torch.cat(self.cache_data.shard, dim=0)
            length = t.shape[0]
        else: 
            t = self.cache_data.shard.copy()
            length = len(t)
        output_path = Path(
            self.cache_data.output_dir, 
            f'shard_{self.cache_data.shard_count}.pt')
        torch.save(t, output_path)
        
        end = self.cache_data.current_start + length
        self.cache_data.manifest['shards'].append({
            'index': self.cache_data.shard_count,
            'filepath': output_path.as_posix(),
            'length': length,
            'start': self.cache_data.current_start,
            'end': end,
            'file_names': self.cache_data.file_names.copy()})
        self.cache_data.total_length += length
        self.cache_data.current_start = end
        self.cache_data.shard.clear()
        self.cache_data.file_names.clear()

    def _init_cache_output(self) -> bool:
        '''Initializes and/or validates an output directory for the cache.
        
        Returns:
            bool : True if a new directory was created for the cache, False if
                a valid cache directory already existed.
        '''
        split = self.dataroot.name
        self.cache_data.output_dir, new = latent_utils.init_latent_cache(
            dataroot=self.dataroot, cache_name=split)
        return new

    def _build_dataloader(self) -> tuple[DiffusionDataset, DataLoader]:
        '''Build a new DiffusionDataset and a Dataloader to load it.
        
        Returns:
            tuple[DiffusionDataset, DataLoader] : The new Dataloader, and the
                dataset it's loading.
        '''
        print(f'Building duplicate Dataloader...\n'
              f'Batch Size : {self.batch_size}\n'
              f'Shard Size : {self.shard_size}\n')
        dataset = DiffusionDataset(
            root_dir=self.dataroot, load_size=self.input_size,
            is_train=False, cache_builder=True, recurse=True)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return dataset, dataloader

    def _init_manifest(self, shard_size: int) -> None:
        '''Initializes the cache manifest with some initial data.

        Args:
            shard_size : The number of batches being saved per shard file.
        '''
        self.cache_data.manifest = {
            'dataroot': self.dataroot.as_posix(),
            'total_length': 0,
            'shard_size': shard_size,
            'shards': [],
            'file_names': []}
        
    def _iterate_dataloader(
            self, 
            dataset: DiffusionDataset,
            dataloader: DataLoader
        ) -> None:
        '''Iterate dataloader, encode to latent, append to current shard.
        
        Args:
            dataset : The dataset which the current dataloader is using. This
                is really only required when caching with metadata, so it can
                find the correct file name for each tensor is encodes.
            dataloader : The DiffusionDataloader to iterate over.
            batch_size : The batch size to use for the caching operation. This
                override exists to allow a larger batch size for caching than
                the one specified in training.
            shard_size : The number of batches to save per shard file.
            store_file_names : Whether to store the original image file names 
                of the encoded tensors for each shard in the manifest.
        '''
        num_batches = len(dataloader)
        export = lambda x: self._export_shard(stack=not x==1)
        for idx, x in enumerate(dataloader):
            latent_utils.print_progress(idx+1, num_batches)
            x = x.to(self.device, non_blocking=True)
            self.cache_data.shard.append(self.vae.encode_to_latent(x).cpu())
            
            if self.store_file_names:
                file_name = dataset.list_files[idx].stem
                self.cache_data.file_names.append(file_name)

            current_size = len(self.cache_data.shard)
            if current_size == self.shard_size: export(x=self.batch_size)
        if not current_size == 0: export(x=self.batch_size)

    def _save_manifest(self) -> None:
        '''Writes manifest data to a JSON file in the cache directory.'''
        print('Caching complete. Writing manifest...')
        self.cache_data.manifest['total_length'] = self.cache_data.total_length
        with open(Path(self.cache_data.output_dir, 'manifest.json'), 'w') as f:
            f.write(json.dumps(self.cache_data.manifest))

    def validate_cache(
            self, 
            shard_directory: PathLike,
            device: str | None=None
        ) -> None:
        print('Validating latent cache...')

        device = device if not device is None else self.device
        shard_directory = Path(shard_directory)
        dir = shard_directory.as_posix()
        if not shard_directory.exists():
            raise FileNotFoundError(
                f'Unable to locate shard directory at path: {dir}')
        shards = list(shard_directory.glob('*.pt'))
        if len(shards) == 0:
            raise FileNotFoundError(
                f'No shards found in shard directory: {dir}')
        for idx, x in enumerate(shards):
            try: shard = torch.load(x, map_location=device)
            except Exception as e:
                raise RuntimeError(
                    f'Unable to load shard file at path: {shard}') from e
            for idy, y in enumerate(shard): 
                assert torch.isfinite(y).all()

                sys.stdout.write('\x1b[1A')
                sys.stdout.write('\x1b[2K')
                print(f'Validating shard ({idx+1}). Iteration: {idy+1}')
        print('Validation complete. No issues found.')

    def cache(
            self, 
            validate_cache: bool=False
        ) -> None:
        '''Loops through Dataloader, encodes tensors to latent space, exports.

        Args:
            batch_size : The batch size to use when caching the latents.
            shard_size : The number of batches to save per shard.

        Returns:
            Path : The path to the cache output directory.
        '''
        self.cache_data = CacheData()
        new = self._init_cache_output()
        if not new: 
            print('Bypassing caching operation...')
            return self.cache_data.output_dir
        
        B, S = self.batch_size, self.shard_size
        dataset, dataloader = self._build_dataloader()
        self._init_manifest(S)
        self._iterate_dataloader(
            dataset=dataset, dataloader=dataloader)
        self._save_manifest()

        if validate_cache:
            self.validate_cache(shard_directory=self.cache_data.output_dir)

    
