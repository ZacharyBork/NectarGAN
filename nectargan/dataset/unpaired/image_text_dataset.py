import random
from os import PathLike

import torch

import nectargan.dataset.metadata.utils as md_utils
from nectargan.dataset.cache_loading.loaders import CacheLoader

class ImageTextDataset(CacheLoader):
    '''Defines a dataset loader for image-text pair training.'''
    def __init__(
            self,
            shard_directory: PathLike,
            metadata_file: PathLike,
            load_size: int
        ) -> None:
        super().__init__(shard_directory, load_size)
        self.metadata = md_utils.load_metadata_file(metadata_file)

    def _get_caption(self, index: int) -> str:
        file_name = self.cached_shard_info['file_names'][index]
        captions = self.metadata.items[file_name]['captions']
        return random.choice(captions)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        t = super().__getitem__(index)
        caption = self._get_caption(self.current_mapped_index)
        return t, caption


