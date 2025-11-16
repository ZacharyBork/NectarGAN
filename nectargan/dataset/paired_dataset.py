import pathlib
from os import PathLike
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

from nectargan.config import Config
from nectargan.dataset import Augmentations

class PairedDataset(Dataset):
    '''Defines a dataset loader for paired training.
    '''
    def __init__(
            self, 
            config: Config, 
            root_dir: PathLike,
            is_train: bool=True
        ) -> None:
        '''Init funtion for PairedDataset class.

        Args:
            config: The Config object being used for the current training.
            root_dir: Pathlike object point to the dataset root.
        '''
        self.config = config
        self.is_train = is_train
        self.load_size = config.dataloader.load.load_size
        self.list_files = [i for i in pathlib.Path(root_dir).iterdir()]
        self.xform = Augmentations(config=self.config)

    def __len__(self) -> int:
        '''Length method override.
        
        Returns:
            int : The number of files in the dataset.
        '''
        return len(self.list_files)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        '''Gets an item from the dataset and applies associated transforms.
        
        Args:
            index : Index of the file to retrieve.

        Raises:
            RuntimeError : If config.common.direction is invalid.
        '''
        size = self.load_size if self.is_train else self.config.dataloader.load.crop_size
        image = np.array(Image.open( # Get input and resize
            self.list_files[index].as_posix()
        ).resize((size*2, size)))

        direction = self.config.dataloader.direction
        if direction == 'AtoB':
            input_image = image[:, :size, :]
            target_image = image[:, size:, :]
        elif self.config.dataloader.direction == 'BtoA':
            input_image = image[:, size:, :] 
            target_image = image[:, :size, :]
        else: 
            message = (f'Invalid direction{direction}\n'
                       f'Valid directions are AtoB and BtoA')
            raise RuntimeError(message)

        if self.is_train:
            return self.xform.apply_transforms_paired(input_image, target_image)
        else: 
            _input = A.Compose([
            A.Normalize(
                mean=[0.5, 0.5, 0.5], 
                std=[0.5, 0.5, 0.5], 
                max_pixel_value=255.0),
            A.ToTensorV2()
        ])(image=input_image)['image']
        
        _target = A.Compose([
            A.Normalize(
                mean=[0.5, 0.5, 0.5], 
                std=[0.5, 0.5, 0.5], 
                max_pixel_value=255.0),
            A.ToTensorV2()
        ])(image=target_image)['image']
        
        return _input, _target 
