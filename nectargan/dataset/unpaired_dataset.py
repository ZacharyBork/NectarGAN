from os import PathLike
from PIL import Image

import numpy as np
import torch
import albumentations as A

from nectargan.config.config_data import Config
from nectargan.dataset.paired_dataset import PairedDataset

class UnpairedDataset(PairedDataset):
    '''Defines a dataset loader for unpaired training.
    '''
    def __init__(
            self, 
            config: Config, 
            root_dir: PathLike,
            is_train: bool=True
        ) -> None:
        super().__init__(config=config, root_dir=root_dir, is_train=is_train)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        '''Gets an item from the dataset and applies associated transforms.
        
        Args:
            index : Index of the file to retrieve.
        '''
        size = self.load_size if self.is_train \
            else self.config.dataloader.load.crop_size
        image = np.array(Image.open( # Get input and resize
            self.list_files[index].as_posix()
        ).resize((size, size)))

        if self.is_train:
            return self.xform.apply_transforms_unpaired(image)
        else: 
            _input = A.Compose([
            A.Normalize(
                mean=[0.5, 0.5, 0.5], 
                std=[0.5, 0.5, 0.5], 
                max_pixel_value=255.0),
            A.ToTensorV2()
        ])(image=image)['image']
        
        return _input 




