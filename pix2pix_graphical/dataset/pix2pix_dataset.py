from PIL import Image
import numpy as np
import pathlib
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from pix2pix_graphical.config.config_data import Config

class Pix2pixDataset(Dataset):
    def __init__(self, config: Config, root_dir: pathlib.Path):
        self.config = config
        self.load_size = config.dataloader.load_size
        self.crop_size = config.dataloader.crop_size
        self.root_dir = root_dir
        self.list_files = [i for i in root_dir.iterdir()]

        self.both_transform = A.Compose(
            [
                A.RandomCrop(height=self.crop_size, width=self.crop_size), 
                A.HorizontalFlip(p=self.config.dataloader.both_flip_chance),
            ], additional_targets={"image0": "image"},
        )

        self.transform_only_input = A.Compose(
            [
                A.ColorJitter(p=self.config.dataloader.input_colorjitter_chance),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )

        self.transform_only_target = A.Compose(
            [
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        image_path = self.list_files[index]
        image = np.array(Image.open(image_path.as_posix()).resize((self.load_size*2, self.load_size)))

        input_image = target_image = np.array
        if self.config.common.direction == 'AtoB':
            input_image, target_image = image[:, :self.load_size, :], image[:, self.load_size:, :]
        elif self.config.common.direction == 'BtoA':
            input_image, target_image = image[:, self.load_size:, :], image[:, :self.load_size, :]
        else: raise Exception('Invalid direction. Valid directions are AtoB and BtoA')

        augmentations = self.both_transform(image=input_image, image0=target_image)
        input_image, target_image = augmentations['image'], augmentations['image0']

        input_image = self.transform_only_input(image=input_image)['image']
        target_image = self.transform_only_target(image=target_image)['image']

        return input_image, target_image
    
