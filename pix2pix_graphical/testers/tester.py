import random
import pathlib
from os import PathLike

import torch
from torchvision.utils import save_image

from pix2pix_graphical.trainers.trainer import Trainer
from pix2pix_graphical.config.config_manager import ConfigManager
from pix2pix_graphical.models.unet.model import UnetGenerator

class Tester(Trainer):
    def __init__(
            self, 
            config: str | PathLike | ConfigManager | None=None
        ) -> None:
        super().__init__(config=config, quicksetup=False)
        self.init_dataloader()
        self.build_output_directory()
        self.init_generator()

    ### INITIALIZATION ###

    def init_generator(self) -> None:
        '''Initializes generator.'''
        tg = self.config.train.generator # Get config train data
        self.gen = UnetGenerator( # Init Generator
            input_size=self.config.dataloader.load.crop_size, 
            in_channels=self.config.dataloader.load.input_nc,
            upconv_type=tg.upsample_block_type)
        self.gen.to(self.device)  # Cast to current device
        self.load_checkpoint('G', self.gen)
        self.gen.eval() # Switch generator into eval mode
        
    def init_dataloader(self) -> None:
        '''Initializes a dataloader for the test dataset.'''
        self.test_loader = self.build_dataloader('test')

    def _init_test_output_root(self) -> None:
        '''Builds a root output directory, or gets the path to an existing one.
        
        Raises:
            RuntimeError : If unable to create test output root directory.
        '''
        test_dir = pathlib.Path(self.experiment_dir, 'test')
        if not test_dir.exists(): 
            try: test_dir.mkdir()
            except Exception as e:
                message = 'Unable to create test output directory'
                raise RuntimeError(message) from e
        self.output_root = test_dir

    def build_output_directory(self) -> None:
        super().build_output_directory()
        self._init_test_output_root()

    def _build_test_output_directory(self) -> None:
        '''Creates an output directory for test results.

        Raises:
            RuntimeError : If unable to create output directory.
        '''
        existing_dirs = list(self.output_root.glob('test_*'))
        if len(existing_dirs) > 0:
            existing_dirs.sort()
            new_version = 1 + int(existing_dirs[-1].name.split('_')[-1])
            new_name = f'test_{str(new_version).zfill(2)}'
            output_path = pathlib.Path(self.output_root, new_name)
        else: output_path = pathlib.Path(self.output_root, 'test_01')
            
        try: output_path.mkdir()
        except Exception as e:
            raise RuntimeError('Unable to create output directory') from e
        self.output_dir = output_path

    def run_test(self, image_count: int=10) -> None:
        self._build_test_output_directory()

        test_data = list(self.test_loader.dataset)
        indices = random.sample(range(len(test_data)), image_count)
        for i, idx in enumerate(indices):
            x, y = test_data[idx]
            x = x.unsqueeze(0).to(self.device)
            y = y.unsqueeze(0).to(self.device)

            with torch.no_grad():
                y_fake = self.gen(x)
            
            imgs = [(x, 'A_real'), (y, 'B_real'), (y_fake, 'B_fake')]
            for img, id in imgs:
                filename = f'{i+1}_{id}.png'
                output_path = pathlib.Path(self.output_dir, filename)
                save_image(img * 0.5 + 0.5, output_path.as_posix())
        