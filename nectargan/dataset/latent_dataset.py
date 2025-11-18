import json
from os import PathLike
from pathlib import Path
from typing import Any

import torch

from nectargan.config import Config
from nectargan.dataset import Augmentations

class LatentDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            config: Config,
            shard_directory: PathLike
        ) -> None:
        super(LatentDataset, self).__init__()
        self.config = config
        self.device = config.common.device
        self.shard_directory = shard_directory
        self.cached_shard: torch.Tensor = None
        self.cached_shard_info: dict[str, Any] = None
        self.xform = Augmentations(config=self.config)

        self._parse_manifest()

    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, index: int) -> torch.Tensor:
        start_index = self._check_index(index)
        mapped_index = index - start_index
        t = self.cached_shard[mapped_index]
        return t.to(self.device)
        # return self.xform.apply_transforms_unpaired(t)

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
        try: self.cached_shard = torch.load(path, map_location=self.device)
        except Exception as e:
            raise RuntimeError(
                f'Unable to load shard file at path: {path}') from e

    def _check_index(self, index: int) -> None:
        start_index = self.cached_shard_info['start']
        end_index = self.cached_shard_info['end']
        if not (start_index <= index < end_index):
            new_shard = self._get_shard_by_index(index)
            self._cache_shard(new_shard)
            start_index = self.cached_shard_info['start']
        return start_index

