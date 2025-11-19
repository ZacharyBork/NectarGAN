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
            is_train: bool=True
        ) -> None:
        super().__init__(config, root_dir, is_train=is_train)
        model_cfg = self.config.model
        match model_cfg.model_type:
            case 'pixel': self.load_size = model_cfg.pixel.input_size
            case 'latent': self.load_size = model_cfg.latent.input_size
            case _: return ValueError(
                f'Invalid model_type: {model_cfg.model_type}')
            
    def __getitem__(self, index: int) -> torch.Tensor:
        '''Gets an item from the dataset.
        
        Args:
            index : Index of the file to retrieve.
        '''
        size = self.load_size
        image = self.load_image_file(index, (size, size))

        _input = A.Compose([
            A.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], 
                max_pixel_value=255.0),
            A.ToTensorV2()
        ])(image=image)['image']
        
        return _input 


