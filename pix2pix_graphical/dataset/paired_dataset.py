import pathlib
from os import PathLike
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset

from pix2pix_graphical.config.config_data import Config
from pix2pix_graphical.dataset.transformer import Transformer

class PairedDataset(Dataset):
    '''Defines a dataset loader for paired training.
    '''
    def __init__(
            self, 
            config: Config, 
            root_dir: PathLike
        ) -> None:
        '''Init funtion for PairedDataset class.

        Args:
            config: The Config object being used for the current training.
            root_dir: Pathlike object point to the dataset root.
        '''
        self.config = config
        self.load_size = config.dataloader.load.load_size
        self.list_files = [i for i in pathlib.Path(root_dir).iterdir()]
        self.xform = Transformer(config=self.config)

    def __len__(self) -> int:
        '''Length method override.
        
        Returns:
            int : The number of files in the dataset.
        '''
        return len(self.list_files)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        '''Gets an item from the dataset and applies associated transforms.
        
        Args:
            index : Index of the file to retrieve.

        Raises:
            RuntimeError : If config.common.direction is invalid.
        '''
        image = np.array(Image.open( # Get input and resize
            self.list_files[index].as_posix()
        ).resize((self.load_size*2, self.load_size)))

        if self.config.dataloader.direction == 'AtoB':
            input_image = image[:, :self.load_size, :]
            target_image = image[:, self.load_size:, :]
        elif self.config.dataloader.direction == 'BtoA':
            input_image = image[:, self.load_size:, :] 
            target_image = image[:, :self.load_size, :]
        else: 
            message = 'Invalid direction. Valid directions are AtoB and BtoA'
            raise RuntimeError(message)

        return self.xform.apply_transforms(input_image, target_image)

    
