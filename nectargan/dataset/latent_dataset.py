import json
import random
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torchvision.transforms import RandomCrop

class LatentShardHotloader(torch.utils.data.Dataset):
    def __init__(
            self, 
            shard_directory: PathLike,
            latent_size: int,
            lazy_load: bool=False,
            silent: bool=False
        ) -> None:
        super(LatentShardHotloader, self).__init__()
        self.latent_size = latent_size
        self.silent = silent
        self.initialized = False
        
        self.lazy_load = lazy_load 
        self.shard_index = 0
        self.tensor_index = 0
        self.length = 0

        self.cache: list[torch.Tensor] = []
        self._get_shard_files(Path(shard_directory))
        self._cache_shards()
        
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, index: int) -> torch.Tensor:
        if self.lazy_load:
            self._update_lazy_loader()
            index = self.tensor_index
        t = self.cache[index]
        if t.ndim == 4 and t.shape[0] == 1: t = t.squeeze(0)
        crop = RandomCrop(size=(self.latent_size, self.latent_size))
        return crop(t)
    
    def _print(self, string: str) -> None: 
        if not self.silent: print(string)
    
    def _get_shard_files(self, shard_directory: Path) -> None:
        self.shard_files = list(shard_directory.glob('*.pt'))
        self.shard_count = len(self.shard_files)
        if self.shard_count == 0:
            raise FileNotFoundError(
                f'Unable to locate shard files at path: '
                f'{shard_directory.as_posix()}')
        
    def _update_lazy_loader(self) -> None:
        self.tensor_index += 1
        if self.tensor_index > len(self.cache):
            self.shard_index = self.shard_index + 1 % self.shard_count
            self._load_shard(self.shard_files[self.shard_index])
            self.tensor_index = 0
        
    def _load_shard(self, shard: Path) -> list[torch.Tensor]:
        if self.initialized: self._print(f'Loading shard: {shard.as_posix()}')
        try: tensors: torch.Tensor = torch.load(shard)
        except Exception as e:
            raise RuntimeError(
                f'Unable to load shard at path: {shard.as_posix()}') from e
        if isinstance(tensors, list): tensors = [i.cpu() for i in tensors]
        else: tensors = list(tensors.split(1, dim=0))
        return tensors
        
    def _cache_shards(self) -> None:
        self._print('Caching shards to memory...')
        self._print(f'Found shards: {self.shard_count}')
        for idx, shard in enumerate(self.shard_files):
            print(f'Caching shard: {idx+1} / {self.shard_count}')
            try: tensors: torch.Tensor = torch.load(shard)
            except Exception as e:
                raise RuntimeError(
                    f'Unable to load shard at path: {shard.as_posix()}') from e
            if isinstance(tensors, list): tensors = [i.cpu() for i in tensors]
            else: tensors = list(tensors.split(1, dim=0))
            
            self.length += len(tensors)
            if not self.lazy_load: self.cache.extend(tensors)
            elif idx == 0: self.cache = tensors
        random.shuffle(self.cache)
        self._print('Caching complete!')
        self.initialized = True

class LatentDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            shard_directory: PathLike,
            latent_size: int
        ) -> None:
        super(LatentDataset, self).__init__()
        self.latent_size = latent_size
        self.shard_directory = shard_directory
        self.cached_shard: torch.Tensor = None
        self.cached_shard_info: dict[str, Any] = None
        
        self._parse_manifest()
        
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, index: int) -> torch.Tensor:
        start_index = self._check_index(index)
        self.current_mapped_index = index - start_index
        t = self.cached_shard[self.current_mapped_index]
        if t.ndim == 4 and t.shape[0] == 1: t = t.squeeze(0)
        crop = RandomCrop(size=(self.latent_size, self.latent_size))
        return crop(t)

    def _parse_manifest(self) -> None:
        manifest = Path(self.shard_directory, 'manifest.json')
        if not manifest.exists():
            raise FileNotFoundError(
                f'Unable to locate manifest at path: {manifest.as_posix()}')
        with open(manifest, 'r') as f: data = json.loads(f.read())
        self.length = data['total_length']
        self.shard_size = data['shard_size']
        self.shards = data['shards']
        self.shard_count = len(self.shards)
        self.indices = [(x['start'], x['end']) for x in self.shards]
        self._cache_shard(self.shards[0])
        
    def _get_shard_by_index(self, index: int) -> dict[str, Any]:
        for idx, x in enumerate(self.indices):
            if (x[0] <= index < x[1]): return self.shards[idx]
            
    def _cache_shard(self, shard: dict[str, Any]) -> None:
        self.cached_shard_info = shard
        path = shard['filepath']
        try: self.cached_shard = torch.load(path)
        except Exception as e:
            raise RuntimeError(
                f'Unable to load shard file at path: {path}') from e
        if isinstance(self.cached_shard, list):
            self.cached_shard = [i.cpu() for i in self.cached_shard]
        else: self.cached_shard = self.cached_shard.cpu()

    def _check_index(self, index: int) -> None:
        start_index = self.cached_shard_info['start']
        end_index = self.cached_shard_info['end']
        if not (start_index <= index < end_index):
            new_shard = self._get_shard_by_index(index)
            self._cache_shard(new_shard)
            start_index = self.cached_shard_info['start']
        return start_index

