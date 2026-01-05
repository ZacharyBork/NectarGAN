from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import albumentations as A

from nectargan.config import Config
from nectargan.dataset import Augmentations, BaseDataset

class UnpairedMaskDataset(BaseDataset):
    def __init__(
            self, 
            config: Config, 
            dataset_directory: PathLike,
            mask_directory: PathLike,
            load_size: int,
            primary_channel: Literal[0, 1, 2]=0,
            secondary_channel: Literal[0, 1, 2]=1,
            processing_operation: int=2,
            blend_strength: float=0.5,
            is_train: bool=True
        ) -> None:
        super().__init__(
            config=config, root_dir=dataset_directory, is_train=is_train)
        self.mask_directory = Path(mask_directory)
        self.load_size = load_size
        self.primary_channel = primary_channel
        self.secondary_channel = secondary_channel
        self.processing_operation = processing_operation
        self.blend_strength = blend_strength
        self.xform = Augmentations(config=self.config)

    def _load_mask_file(self, index) -> np.ndarray:
        current_image = Path(self.list_files[index])
        mask_file = Path(self.mask_directory, current_image.name)
        mask = self.load_image_file(
            index, self.load_size, preserve_aspect_ratio=True,
            image_file=mask_file)
        return mask
    
    def _process_mask(self, mask: np.ndarray) -> np.ndarray:
        '''
        Operations:
            0 : Black mask
            1 : White mask
            2 : Binary channel mask (from primary_channel)
            3 : Binary subtraction (primary_channel - secondary_channel)
            4 : Blended subtraction (prim_ch - sec_ch * (1.0 - blend_strength))
            5 : Maximum (max(prim_ch, sec_ch * blend_strength))
        '''
        M = mask
        match self.processing_operation:
            case 0: result = np.zeros(M.shape[:2], dtype=np.float32)
            case 1: result = np.ones(M.shape[:2], dtype=np.float32)
            case 2:
                P = M[:, :, self.primary_channel] > 128
                result = P.astype(np.float32)
            case 3:
                P = M[:, :, self.primary_channel] > 128
                S = M[:, :, self.secondary_channel] > 128
                result = (P & ~S).astype(np.float32)
            case 4:
                P = M[:, :, self.primary_channel].astype(np.float32) / 255.0
                S = M[:, :, self.secondary_channel].astype(np.float32) / 255.0
                result = P - ((1.0 - self.blend_strength) * S)
                result = np.clip(result, 0, 1)
            case 5:
                P = M[:, :, self.primary_channel].astype(np.float32) / 255.0
                S = M[:, :, self.secondary_channel].astype(np.float32) / 255.0
                result = np.maximum(P, self.blend_strength * S)
                result = np.clip(result, 0, 1)

    def __getitem__(
            self, 
            index: int
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
        image = self.load_image_file(
            index, self.load_size, preserve_aspect_ratio=True)
        
        if self.is_train:
            mask = self._load_mask_file()
            mask = self._process_mask(mask)
            _i, _m, = self.xform.apply_masked_transforms_unpaired(image, mask)           
            return (_i, _m)
        else: 
            mean = std = [0.5, 0.5, 0.5]
            _image = A.Compose([
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                A.ToTensorV2()
            ])(image=image)['image']
            return _image, None

