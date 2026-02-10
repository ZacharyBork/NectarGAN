import json
import random
from os import PathLike
from pathlib import Path
from typing import Any, Literal
from dataclasses import dataclass, field

import torch
from torchvision.transforms import RandomCrop, CenterCrop, Resize

class CacheLoader(torch.utils.data.Dataset):
    def __init__(
            self, 
            shard_directory: PathLike,
            load_size: int,
            crop_type: Literal['random', 'center'] | None = 'center',
        ) -> None:
        super(CacheLoader, self).__init__()
        self.shard_directory = shard_directory
        self.load_size = load_size
        self.crop_type = crop_type

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
        crop = RandomCrop(size=(self.load_size, self.load_size))
        return self._crop_tensor(t)

    def _crop_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        size = (self.load_size, self.load_size)
        match self.crop_type:
            case 'random': crop = RandomCrop(size=size)
            case 'center': crop = CenterCrop(size=size)
            case None:     crop = Resize(size=size)
            case _: raise ValueError(f'Invalid crop type: {self.crop_type}')
        return crop(tensor)

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

class CacheShardHotloader(torch.utils.data.Dataset):
    def __init__(
            self, 
            shard_directory: PathLike,
            load_size: int,
            crop_type: Literal['random', 'center'] | None = 'center',
            lazy_load: bool = False,
            periodic_recache: bool = False,
            recache_period: int = 50000,
            silent: bool = False
        ) -> None:
        super(CacheShardHotloader, self).__init__()
        self.shard_directory = shard_directory
        self.load_size = load_size
        self.crop_type = crop_type
        self.periodic_recache = periodic_recache
        self.recache_period = recache_period
        self.silent = silent

        self.lazy_load = lazy_load 
        self.shard_index = 0
        self.tensor_index = 0
        
        self.initialized = False
        self.load_counter = 0
        self.length = 0

        self.cache: list[torch.Tensor] = []
        self._get_shard_files(Path(self.shard_directory))
        self._cache_shards()

    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, index: int) -> torch.Tensor:
        self.load_counter += 1
        if self.periodic_recache:
            if self.load_counter % self.recache_period == 0:
                self.shard_index = 0
                self.tensor_index = 0
                self._get_shard_files(Path(self.shard_directory))
                self._cache_shards()
        if self.lazy_load:
            self._update_lazy_loader()
            index = self.tensor_index
        t = self.cache[index]
        if t.ndim == 4 and t.shape[0] == 1: t = t.squeeze(0)
        
        return self._crop_tensor(t) 
    
    def _print(self, string: str) -> None: 
        if not self.silent: print(string)
    
    def _crop_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        size = (self.load_size, self.load_size)
        match self.crop_type:
            case 'random': crop = RandomCrop(size=size)
            case 'center': crop = CenterCrop(size=size)
            case None:     crop = Resize(size=size)
            case _: raise ValueError(f'Invalid crop type: {self.crop_type}')
        return crop(tensor)

    def _get_shard_files(self, shard_directory: Path) -> None:
        self.shard_files = list(shard_directory.glob('*.pt'))
        self.shard_count = len(self.shard_files)
        if self.shard_count == 0:
            raise FileNotFoundError(
                f'Unable to locate shard files at path: '
                f'{shard_directory.as_posix()}')
        
    def _shard_to_list(self, shard: torch.Tensor) -> list[torch.Tensor]:
        if isinstance(shard, list): return [i.cpu() for i in shard]
        else: return list(shard.split(1, dim=0))

    def _load_shard(self, shard: Path) -> list[torch.Tensor]:
        if self.initialized: self._print(f'Loading shard: {shard.as_posix()}')
        try: tensors: torch.Tensor = torch.load(shard)
        except Exception as e:
            raise RuntimeError(
                f'Unable to load shard at path: {shard.as_posix()}') from e
        return self._shard_to_list(tensors)

    def _update_lazy_loader(self) -> None:
        self.tensor_index += 1
        if self.tensor_index > len(self.cache):
            self.cache.clear()
            self.tensor_index = 0
            self.shard_index = self.shard_index + 1 % self.shard_count
            self.cache = self._load_shard(self.shard_files[self.shard_index])
 
    def _cache_shards(self) -> None:
        self._print('Caching shards to memory...')
        self._print(f'Found shards: {self.shard_count}')
        for idx, shard in enumerate(self.shard_files):
            print(f'Caching shard: {idx+1} / {self.shard_count}')
            tensors = self._load_shard(shard)
            self.length += len(tensors)
            
            if not self.lazy_load: self.cache.extend(tensors)
            elif idx == 0: self.cache = tensors
        random.shuffle(self.cache)
        self._print('Caching complete!')
        self.initialized = True



@dataclass
class MixerShardData:
    root: Path
    mix_amount: float
    shard_count: int
    shards: list[Path]
    current_shard_index: int
    buffer: list[torch.Tensor] = field(default_factory=list)

class CacheMixer(CacheShardHotloader):

    # CURRENTLY NOT FUNCTIONAL!

    # Structure of caches input argument:
    #
    # caches = {
    #     '/path/to/cache/directory1': 0.5 <- Mix percentage
    #     '/path/to/cache/directory2': 0.3
    #     '/path/to/cache/directory3': 0.2
    #  }

    def __init__(
            self, 
            caches: dict[str, float],
            load_size: int,
            crop_type: Literal['random', 'center'] | None = 'center',
            silent: bool = False
        ) -> None:
        torch.utils.data.Dataset.__init__(self)
        self.weights: list[float] = []
        self.load_size = load_size
        self.crop_type = crop_type
        self.silent = silent

        self.length = 999999999
        self.initialized = False
    
        self.caches = self._init_caches(caches)
        self._normalize_weights()

    def __getitem__(self, index: int) -> torch.Tensor:
        cache = random.choices(self.caches, weights=self.weights, k=1)[0]
        buffer = cache.buffer
        t = buffer[0]
        buffer.pop(0)
        if len(buffer) == 0: self._buffer_next_shard(cache)
        if t.ndim == 4 and t.shape[0] == 1: t = t.squeeze(0)
        return self._crop_tensor(t) 

    def _buffer_next_shard(self, shard_data: MixerShardData) -> None:
        index = (shard_data.current_shard_index + 1) % shard_data.shard_count
        shard = shard_data.shards[index]
        shard_data.buffer = self._load_shard(shard)
        shard_data.current_shard_index = index

    def _init_caches(self, caches: dict[str, float]) -> list[MixerShardData]:
        output = []
        for key, value in caches.items():
            cache_directory = Path(key)
            if not cache_directory.exists():
                raise FileNotFoundError(
                    f'Unable to locate cache directory at path: '
                    f'{cache_directory.as_posix()}')
            shards = list(cache_directory.glob('*.pt'))
            count = len(shards)
            if count == 0:
                raise FileNotFoundError(
                    f'Unable to locate shard files in cache directory: '
                    f'{cache_directory.as_posix()}')
            
            data = MixerShardData(
                root=cache_directory, shard_count=count, shards=shards,
                current_shard_index=0, buffer=self._load_shard(shards[0]))
            output.append(data)
            self.weights.append(value)
        self.initialized = True
        return output
    
    def _normalize_weights(self) -> None:
        total = sum(self.weights)
        self.weights = [i / max(1e-6, total) for i in self.weights]




