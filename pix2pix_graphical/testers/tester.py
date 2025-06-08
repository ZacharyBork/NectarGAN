import json
import random
import pathlib
from os import PathLike
from typing import Any

import torch
from torchvision.utils import save_image

from pix2pix_graphical.trainers.trainer import Trainer
from pix2pix_graphical.config.config_manager import ConfigManager
from pix2pix_graphical.models.unet.model import UnetGenerator
from pix2pix_graphical.losses import losses

class Tester(Trainer):
    def __init__(
            self, 
            config: str | PathLike | ConfigManager | None=None
        ) -> None:
        super().__init__(config=config, quicksetup=False)
        self.config.train.load.continue_train = True
        self.init_dataloader()
        self.build_output_directory()
        self.init_generator()
        self._init_loss_functions()

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

    def _init_loss_functions(self) -> None:
        self.L1 = torch.nn.L1Loss().to(self.device)
        self.SOBEL = losses.Sobel()
        self.LAPLACIAN = losses.Laplacian()
        
    def init_dataloader(self) -> None:
        '''Initializes a dataloader for the test dataset.'''
        self.test_loader = self.build_dataloader('test')

    ### OUTPUT HANDLING ###

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
            self.test_version = 1 + int(existing_dirs[-1].name.split('_')[-1])
        else: self.test_version = 1
        new_name = f'test_{str(self.test_version).zfill(2)}'
        output_path = pathlib.Path(self.output_root, new_name)
            
        try: output_path.mkdir()
        except Exception as e:
            raise RuntimeError('Unable to create output directory') from e
        self.output_dir = output_path

    ### LOGGING ###

    def _export_base_test_log(self) -> None:
        self.test_log = pathlib.Path(self.output_dir, 'log.json')
        structure = {
            'experiment': {
                'name': self.config.common.experiment_name,
                'version': self.config.common.experiment_version,
                'path': self.experiment_dir.as_posix()
            },
            'test': {
                'version': self.test_version,
                'path': self.output_dir.as_posix(),
                'results': []
            }
        }
        try: 
            with open(self.test_log, 'w') as file:
                file.write(json.dumps(structure, indent=4))
        except Exception as e:
            raise RuntimeError('Unable to export base test log.') from e
        
    def _write_log_entry(self, entry: dict[str, Any]) -> None:
        with open(self.test_log, 'r') as file:
            log_data = json.loads(file.read())
        log_data['test']['results'].append(entry)
        with open(self.test_log, 'w') as file:
            file.write(json.dumps(log_data, indent=4))

    ### RUN TEST ###

    def run_test(self, image_count: int=10) -> None:
        self._build_test_output_directory()
        self._export_base_test_log()

        test_data = list(self.test_loader.dataset)
        indices = random.sample(range(len(test_data)), image_count)
        for i, idx in enumerate(indices):
            print(f'Running test iteration {i+1}', flush=True)

            x, y = test_data[i]
            x = x.unsqueeze(0).to(self.device)
            y = y.unsqueeze(0).to(self.device)

            with torch.no_grad():
                y_fake = self.gen(x)
            
            imgs = [(x, 'A_real'), (y, 'B_real'), (y_fake, 'B_fake')]
            for img, id in imgs:
                filename = f'{i+1}_{id}.png'
                output_path = pathlib.Path(self.output_dir, filename)
                save_image(img * 0.5 + 0.5, output_path.as_posix())

            y = y.detach().cpu()
            y_fake = y_fake.detach().cpu()

            l1 = self.L1(y, y_fake)
            sobel = self.SOBEL(y, y_fake)
            laplacian = self.LAPLACIAN(y, y_fake)
            image_path = self.test_loader.dataset.list_files[idx].as_posix()
            log_entry = {
                'iteration': idx,
                'test_data_path': image_path,
                'losses': {
                    'L1': l1.mean().item(),
                    'SOBEL': sobel.mean().item(),
                    'LAPLACIAN': laplacian.mean().item()
                } 
            }
            self._write_log_entry(entry=log_entry)
        