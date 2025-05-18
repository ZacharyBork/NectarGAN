from PIL import Image
import numpy as np
import pathlib
from torch.utils.data import Dataset

from .. import config

class Pix2pixDataset(Dataset):
    def __init__(self, root_dir):
        self.input_resolution = 512
        self.root_dir = pathlib.Path(root_dir)
        self.list_files = self.root_dir.iterdir()

    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        image_path = self.list_files[index]
        image = np.array(Image.open(image_path.as_posix()))
        input_image = image[:, :self.input_resolution, :]
        target_image = image[:, :self.input_resolution, :]

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image, target_image = augmentations['image'], augmentations['image0']

        input_image = config.transform_only_input(image=input_image)['image']
        target_image = config.transform_only_target(image=target_image)['image']

        return input_image, target_image