import torch
import numpy as np
import albumentations as A
from pix2pix_graphical.config.config_data import Config

class Transformer():
    '''Manages dataset transforms.
    '''
    def __init__(
            self, 
            config: Config,
            mean: list[float]=[0.5, 0.5, 0.5],
            std: list[float]=[0.5, 0.5, 0.5],
            max_pixel_value: float=255.0
        ) -> None:
        '''Init function for Transform manager class.

        Args:
            config : The Config object being used for the current training.
            mean : Mean value for normalization.
            std : Standard deviation for normalization.
            max_pixel_value : Maximum pixel value for normalization.
        '''
        self.augs = config.dataloader.augmentations
        self.crop_size = config.dataloader.crop_size
        
        self.mean, self.std = mean, std
        self.max_value = max_pixel_value

        # Init transform functions
        self.transform_both = self._both_transform()
        self.transform_input = self._input_transform()
        self.transform_target = self._target_transform()

    def _append_xform_by_value(
            self, 
            xform: torch.nn.Module, 
            seq: list[torch.nn.Module],
            value: float
        ) -> None:
        '''Appends xform to a seq (or not), based on given value being > 0.0.

        This is used to append transforms to a sequence based on the associated
        chance value in the config. So that if the chance of applying the
        transform is 0.0, we don't bother adding it to the Compose.

        Args:
            xform : The transform to append if value > 0.0.
            seq : The sequence (list[torch.nn.Module]) to append it to.
            value : The value to check, generally a random chance from config.
        '''
        if value > 0.0: seq.append(xform(p=value))

    def _both_transform(self) -> A.Compose:
        '''Builds transform function that is applied to both input and target.
        '''
        b = self.augs.both
        xforms = [A.RandomCrop(height=self.crop_size, width=self.crop_size)]
        
        self._append_xform_by_value(
            A.HorizontalFlip,
            seq=xforms, value=b.h_flip_chance)
        self._append_xform_by_value(
            A.VerticalFlip,
            seq=xforms, value=b.v_flip_chance)
        self._append_xform_by_value(
            A.RandomRotate90,
            seq=xforms, value=b.rot90_chance)
        
        return A.Compose(xforms, additional_targets={ 'image0': 'image' })

    def _input_transform(self) -> A.Compose:
        '''Builds transform function that is applied only to input.
        '''
        return A.Compose([
            A.ColorJitter(
                (self.augs.input.colorjitter_brightness[0], 
                self.augs.input.colorjitter_brightness[1]), 
                p=self.augs.input.colorjitter_chance),
            A.Normalize(
                mean=self.mean, 
                std=self.std, 
                max_pixel_value=self.max_value),
            A.ToTensorV2(),
        ]) 

    def _target_transform(self) -> A.Compose:
        '''Builds transform function that is applied only to target.
        '''
        return A.Compose([
            A.Normalize(
                mean=self.mean, 
                std=self.std, 
                max_pixel_value=self.max_value),
            A.ToTensorV2(),
        ]) 

    def apply_transforms(
            self, 
            input_image: np.ndarray, 
            target_image: np.ndarray
        ) -> tuple[torch.Tensor, torch.Tensor]:
        '''Applies transforms to input and target image and returns results.

        Args:
            input_image : The input image of the current dataset pair.
            target_image : The target image of the current dataset pair.
        '''
        aug = self.transform_both(image=input_image, image0=target_image)
        
        _input, _target = aug['image'], aug['image0']
        
        _input = self.transform_input(image=_input)['image']
        _target = self.transform_target(image=_target)['image']

        return _input, _target