from os import PathLike

import numpy as np
import torch
import albumentations as A

from nectargan.config import GANConfig
from nectargan.dataset import Augmentations, BaseDataset

class PairedDataset(BaseDataset[GANConfig]):
    '''Defines a dataset loader for paired training.
    '''
    def __init__(
            self, 
            config: GANConfig, 
            root_dir: PathLike,
            is_train: bool=True
        ) -> None:
        '''Init funtion for PairedDataset class.

        Args:
            config: The Config object being used for the current training.
            root_dir: Pathlike object point to the dataset root.
        '''
        super().__init__(
            config=config, root_dir=root_dir, is_train=is_train)
        self.load_size = config.dataloader.load.load_size
        self.xform = Augmentations(config=self.config)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        '''Gets an item from the dataset and applies associated transforms.
        
        Args:
            index : Index of the file to retrieve.

        Raises:
            RuntimeError : If config.common.direction is invalid.
        '''
        size = self.load_size if self.is_train \
            else self.config.dataloader.load.crop_size
        image = self.load_image_file(index, size, preserve_aspect_ratio=True)

        input_img, target_img = self.split_images(image, size)

        if self.is_train:
            return self.xform.apply_transforms_paired(input_img, target_img)
        else: 
            mean = std = [0.5, 0.5, 0.5]
            _input = A.Compose([
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                A.ToTensorV2()
            ])(image=input_img)['image']
            
            _target = A.Compose([
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                A.ToTensorV2()
            ])(image=target_img)['image']
            return _input, _target 

    def split_images(self, image: np.ndarray, size: int) -> tuple[np.ndarray]:
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
        return input_image, target_image
