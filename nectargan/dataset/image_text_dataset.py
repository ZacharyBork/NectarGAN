import json
import random
from pathlib import Path
from os import PathLike
from typing import Any
from dataclasses import dataclass

import torch

from nectargan.config import DiffusionConfig
from nectargan.dataset import LatentDataset

@dataclass
class TextEmbeddedMetadata:
    schema_version: int
    total_captions: int
    total_images:   int
    items: dict[str, dict[str, Any]]

class ImageTextDataset(LatentDataset):
    '''Defines a dataset loader for image-text pair training.'''
    def __init__(
            self,
            config: DiffusionConfig,
            shard_directory: PathLike,
            metadata_file: PathLike,
            latent_size: int
        ) -> None:
        super().__init__(config, shard_directory, latent_size)
        self.metadata: TextEmbeddedMetadata = None
        self._load_metadata_file(metadata_file)

    def _load_metadata_file(self, metadata_file: PathLike) -> None:
        self.metadata_file = Path(metadata_file)
        if not self.metadata_file.exists():
            raise FileNotFoundError(
                f'Unable to locate metadata file at path: '
                f'{self.metadata_file.as_posix()}')
        with open(self.metadata_file, 'r') as file:
            metadata = json.loads(file.read())
        self.metadata = TextEmbeddedMetadata(
            schema_version=metadata['info']['schema_version'],
            total_captions=metadata['info']['total_captions'],
            total_images=metadata['info']['total_images'],
            items=metadata['items'])

    def _get_caption(self, index: int) -> str:
        file_name = self.cached_shard_info['file_names'][index]
        captions = self.metadata.items[file_name]['captions']
        return random.choice(captions)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        t = super().__getitem__(index)
        caption = self._get_caption(self.current_mapped_index)
        return t, caption


