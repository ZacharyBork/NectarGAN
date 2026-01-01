import random
from os import PathLike

import torch
import albumentations as A

import nectargan.dataset.metadata.utils as md_utils
from nectargan.config import DiffusionConfig
from nectargan.dataset import BaseDataset

class DiffusionDataset(BaseDataset[DiffusionConfig]):
    '''Defines a dataset loader for unpaired training.'''
    def __init__(
            self, 
            root_dir: PathLike,
            load_size: int,
            metadata_file: PathLike | None=None,
            is_train: bool=True,
            cache_builder: bool=False,
            recurse: bool=False,
            recurse_for_type: str='jpg'
        ) -> None:
        super().__init__(
            None, root_dir, is_train=is_train, 
            recurse=recurse, recurse_for_type=recurse_for_type)
        self.metadata = md_utils.load_metadata_file(metadata_file) \
            if not metadata_file is None else None 
        self.cache_builder = cache_builder
        self.load_size = load_size
            
    def _get_caption(self, index: int) -> str:
        file_name = self.list_files[index].stem
        captions = self.metadata.items[file_name]['captions']
        return random.choice(captions)

    def __getitem__(
            self, 
            index: int
        ) -> torch.Tensor | tuple[torch.Tensor, str | None]:
        '''Gets an item from the dataset.
        
        Args:
            index : Index of the file to retrieve.
        '''
        image = self.load_image_file(
            index, self.load_size, preserve_aspect_ratio=True, to_rgb=True)
        
        transforms = [
            A.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], 
                max_pixel_value=255.0),
            A.ToTensorV2()]
        if not self.cache_builder: 
            transforms.insert(0, A.RandomCrop(self.load_size, self.load_size))
            _image = A.Compose(transforms)(image=image)['image']
            caption = self._get_caption(index) \
                if not self.metadata is None else None
            return _image, caption
        else:
            _image = A.Compose(transforms)(image=image)['image']
            return _image

