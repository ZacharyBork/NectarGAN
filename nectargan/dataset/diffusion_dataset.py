from os import PathLike

import torch
import albumentations as A

from nectargan.config import DiffusionConfig
from nectargan.dataset import BaseDataset

class DiffusionDataset(BaseDataset[DiffusionConfig]):
    '''Defines a dataset loader for unpaired training.'''
    def __init__(
            self, 
            config: DiffusionConfig, 
            root_dir: PathLike,
            is_train: bool=True,
            cache_builder: bool=False
        ) -> None:
        super().__init__(config, root_dir, is_train=is_train)
        self.cache_builder = cache_builder
        cfg = self.config.model
        match cfg.model_type:
            case 'pixel': self.load_size = cfg.pixel.input_size
            case 'latent': self.load_size = cfg.latent.input_size
            case 'stable': self.load_size = cfg.stable.input_size
            case _: return ValueError(
                f'Invalid model_type: {cfg.model_type}')
            
    def __getitem__(self, index: int) -> torch.Tensor:
        '''Gets an item from the dataset.
        
        Args:
            index : Index of the file to retrieve.
        '''
        image = self.load_image_file(
            index, self.load_size, preserve_aspect_ratio=True, to_rgb=True)

        mean = [0.5, 0.5, 0.5]
        _input = A.Compose([
            A.Normalize(mean=mean, std=mean, max_pixel_value=255.0),
            A.ToTensorV2()
        ])(image=image)['image']
        
        return _input 


