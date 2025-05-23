from PIL import Image
import numpy as np
import pathlib
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Pix2pixDataset(Dataset):
    def __init__(self, config: dict, root_dir: pathlib.Path):
        self.config = config
        self.root_dir = root_dir
        self.list_files = [i for i in root_dir.iterdir()]

        self.both_transform = A.Compose(
            [
                # A.Resize(width=self.config['LOAD_SIZE'], height=self.config['LOAD_SIZE']),
                # A.RandomCrop(height=self.config['CROP_SIZE'], width=self.config['CROP_SIZE']), 
                A.Resize(width=self.config['COMMON']['CROP_SIZE'], height=self.config['COMMON']['CROP_SIZE']),
                A.HorizontalFlip(p=0.5),
            ], additional_targets={"image0": "image"},
        )

        self.transform_only_input = A.Compose(
            [
                A.ColorJitter(p=0.1),
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
        image = np.array(Image.open(image_path.as_posix()))

        input_image = target_image = np.array
        if self.config['COMMON']['DIRECTION'] == 'AtoB':
            input_image, target_image = image[:, :self.config['COMMON']['LOAD_SIZE'], :], image[:, self.config['COMMON']['LOAD_SIZE']:, :]
        elif self.config['COMMON']['DIRECTION'] == 'BtoA':
            input_image, target_image = image[:, self.config['COMMON']['LOAD_SIZE']:, :], image[:, :self.config['COMMON']['LOAD_SIZE'], :]
        else: raise Exception('Invalid direction. Valid directions are AtoB and BtoA')

        augmentations = self.both_transform(image=input_image, image0=target_image)
        input_image, target_image = augmentations['image'], augmentations['image0']

        input_image = self.transform_only_input(image=input_image)['image']
        target_image = self.transform_only_target(image=target_image)['image']

        return input_image, target_image
    
