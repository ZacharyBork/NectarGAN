from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import albumentations as A

from nectargan.config import GANConfig
from nectargan.dataset import PairedDataset

class PairedMaskDataset(PairedDataset):
    '''Defines a dataset loader for paired training.
    '''
    def __init__(
            self, 
            config: GANConfig, 
            root_dir: PathLike,
            mask_directory: PathLike,
            mask_channel: Literal[0, 1, 2]=0,
            combination_type: Literal[
                'no_mask', 'single_channel', 'boolean_subtract', 
                'blend_subtract', 'maximum', 
            ] = 'single_channel',
            blend_channel: Literal[0, 1, 2]=1,
            blend_amount: float = 0.5,
            is_train: bool=True
        ) -> None:
        super().__init__(config=config, root_dir=root_dir, is_train=is_train)
        self.mask_directory = Path(mask_directory)
        self.mask_channel = mask_channel
        self.combination_type = combination_type
        self.blend_channel = blend_channel
        self.blend_amount = blend_amount

    def __getitem__(
            self, 
            index: int
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        size = self.load_size
        image = self.load_image_file(index, size, preserve_aspect_ratio=True)
        input_img, target_img = self.split_images(image, size)
        
        if self.is_train:
            current_image = Path(self.list_files[index])
            mask_file = Path(self.mask_directory, current_image.name)
            mask = self.load_image_file(
                index, size, preserve_aspect_ratio=True,
                image_file=mask_file)
            
            match self.combination_type:
                case 'no_mask':
                    mask = np.ones_like(mask).astype(np.uint8) * 255
                case 'single_channel': 
                    green = mask[:, :, 1].astype(np.float32)
                    mask = np.stack([green, green, green], axis=2)
                case 'boolean_subtract':
                    red = mask[:, :, 0] > 128
                    green = mask[:, :, 1] > 128
                    mask = (red & ~green).astype(np.uint8) * 255
                    mask = np.stack([mask, mask, mask], axis=2)
                case 'blend_subtract':
                    red = mask[:, :, 0].astype(np.float32)
                    green = mask[:, :, 1].astype(np.float32)
                    result = red - (self.blend_amount * green)
                    result = np.clip(result, 0, 255).astype(np.uint8)
                    mask = np.stack([result, result, result], axis=2)
                case 'maximum':
                    red = mask[:, :, 0].astype(np.float32)
                    green = mask[:, :, 1].astype(np.float32)
                    result = np.maximum(green, (0.3 * red))
                    result = np.clip(result, 0, 255).astype(np.uint8)
                    mask = np.stack([result, result, result], axis=2)

            input_mask, target_mask = self.split_images(mask, size)
            _i, _t, _im, _tm = self.xform.apply_masked_transforms_paired(
                input_img, target_img, input_mask, target_mask)
            
            return (_i, _t, _im, _im)
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
            dummy = torch.zeros_like(_input)
            return _input, _target, dummy, dummy

