import pathlib
from PIL import Image
from os import PathLike
from typing import TypeVar, Generic

import numpy as np
from torch.utils.data import Dataset

from nectargan.config import Config

TConfig = TypeVar('TConfig', bound=Config)

class BaseDataset(Dataset, Generic[TConfig]):
    def __init__(
            self, 
            config: TConfig, 
            root_dir: PathLike,
            is_train: bool=True
        ) -> None:
        self.config: TConfig = config
        self.is_train = is_train
        self.list_files = [i for i in pathlib.Path(root_dir).iterdir()]

    def __len__(self) -> int:
        '''Length method override.
        
        Returns:
            int : The number of files in the dataset.
        '''
        return len(self.list_files)
    
    def __getitem__(self, index: int) -> None:
        raise NotImplementedError(
            'This method is implemented by the child class.')

    def load_image_file(
            self, 
            index: int, 
            size: int,
            preserve_aspect_ratio: bool=False,
            to_rgb: bool=False
        ) -> np.ndarray:
        img = Image.open(self.list_files[index].as_posix())
        if to_rgb: img = img.convert('RGB')
        if preserve_aspect_ratio:
            x = lambda y, z: int(round(y*z))
            scale = size / min(img.width, img.height)
            resolution = (x(img.width, scale), x(img.height, scale))
        else: resolution = (size, size)
        img = img.resize(resolution, resample=Image.Resampling.BICUBIC)
        return np.array(img)



