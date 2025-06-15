import json
import random
import pathlib
from os import PathLike
from typing import Any

import torch
from torchvision.utils import save_image

from nectargan.trainers.trainer import Trainer
from nectargan.config.config_manager import ConfigManager
from nectargan.models.unet.model import UnetGenerator
from nectargan.losses import losses

class Tester(Trainer):
    def __init__(
            self, 
            config: str | PathLike | ConfigManager | None=None,
            experiment_dir: PathLike | None=None,
            dataroot: PathLike | None=None,
            load_epoch: int | None=None
        ) -> None:
        super().__init__(config=config, quicksetup=False)
        self.config.train.load.continue_train = True
        if not load_epoch is None:
            self.config.train.load.load_epoch = load_epoch
        if not experiment_dir is None:
            self.experiment_dir = pathlib.Path(experiment_dir)
            self._init_test_output_root()
        else: self._build_output_directory()
        if not dataroot is None:
            self.config.dataloader.dataroot = dataroot
        self._init_dataloader()
        self._init_generator()
        
    ### INITIALIZATION ###

    def _init_generator(self) -> None:
        '''Initializes generator.'''
        tg = self.config.train.generator # Get config train data
        self.gen = UnetGenerator( # Init Generator
            input_size=self.config.dataloader.load.crop_size, 
            in_channels=self.config.dataloader.load.input_nc,
            features=tg.features,
            n_downs=tg.n_downs,
            block_type=tg.block_type,
            upconv_type=tg.upsample_type)
        self.gen.to(self.device)  # Cast to current device
        self.load_checkpoint('G', self.gen)
        self.gen.eval() # Switch generator into eval mode

    def init_loss_functions(self) -> None:
        '''Inits loss functions so that they can be run during testing.'''
        self.L1 = torch.nn.L1Loss().to(self.device)
        self.SOBEL = losses.Sobel().to(self.device)
        self.LAPLACIAN = losses.Laplacian().to(self.device)
        
    def _init_dataloader(self) -> None:
        '''Initializes a dataloader for the test dataset.'''
        self.test_loader = self.build_dataloader('test')
        self.test_data = list(self.test_loader.dataset)

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

    def _build_output_directory(self) -> None:
        '''Builds a root test output directory inside the experiment direcotry.
        
        This function first calls `Trainer.build_output_directory()`. Since we
        are necessarily loading the weights of a pre-trained model, rather
        than create an output directory, this function as it is used here will
        just assign the correct value to `self.experiment_dir`. Once that is
        assigned, it will generate a new directory called `test` inside of the
        experiment dir in which to generate subdirectories for each test that
        is run.
        '''
        super().build_output_directory()
        self._init_test_output_root()

    def build_test_output_directory(self) -> None:
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

    def export_base_test_log(self) -> None:
        '''Exports a base JSON log to the test output directory.'''
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
        '''Adds an entry to the test log.'''
        with open(self.test_log, 'r') as file:
            log_data = json.loads(file.read())
        log_data['test']['results'].append(entry)
        with open(self.test_log, 'w') as file:
            file.write(json.dumps(log_data, indent=4))

    ### RUN TEST ###

    def run_inference(
            self, 
            x: torch.Tensor,
            y: torch.Tensor
        ) -> dict[str, torch.Tensor]:
        '''Runs model inference and sanitizes [x, y, y_fake] tensors.
        
        Args:
            x : The real input image tensor.
            y : The ground truth output image tensor.

        Returns:
            dict : Dict of [x, y, y_fake] tensors mapped by their name.
        '''
        x = x.unsqueeze(0).to(self.device)
        y = y.unsqueeze(0).to(self.device)

        with torch.no_grad():
            y_fake = self.gen(x)

        return { 'x': x, 'y': y, 'y_fake': y_fake }

    def save_test_images(
            self, 
            index: int, 
            x: torch.Tensor, 
            y: torch.Tensor,
            y_fake: torch.Tensor
        ) -> list[str]:
        '''Exports [x, y, y_fake] images to the test output directory.
        
        Args:
            index : The current test loop index when this function is called.
            x : The real input image tensor.
            y : The ground truth output tensor.
            y_fake : The generated fake output tensor.
        
        Returns:
            list[str] : List containing the system paths of the exported files
                as strings. They are in the order [x, y, y_fake].
        '''
        imgs = [(x, 'A_real'), (y, 'B_real'), (y_fake, 'B_fake')]
        paths = []
        for img, id in imgs:
            filename = f'{index+1}_{id}.png'
            output_path = pathlib.Path(self.output_dir, filename)
            save_image(img * 0.5 + 0.5, output_path.as_posix())
            paths.append(output_path.as_posix())
        return paths

    def run_losses(
            self, 
            y: torch.Tensor, 
            y_fake: torch.Tensor
        ) -> dict[str, float]:
        '''Runs loss functions on generated tensor and returns the mean values.
        
        Args: 
            y : The ground truth image tensor.
            y_fake : The generated fake image tensor.

        Returns:
            dict : A dict of loss values mapped by the corresponding loss name.
        '''
        l1 = self.L1(y, y_fake).mean().item()
        sobel = self.SOBEL(y, y_fake).mean().item()
        laplacian = self.LAPLACIAN(y, y_fake).mean().item()
        return { 'L1': l1, 'SOBEL': sobel, 'LAPLACIAN': laplacian }
    
    def get_current_image_path(self, index: int) -> str:
        '''Gets the path of the image that the model is currently testing on.
        
        Args:
            index : The current test loop index.
        '''
        return self.test_loader.dataset.list_files[index].as_posix()
    
    def build_lookup_indices(self, count: int) -> list[int]:
        '''Builds a list of len(count) of indices for dataset lookup.
        
        Args:
            count : The number of indices to create. If this value is greater
                than the total number of images in the test dataset, the total
                image count will be used instead.
        
        Returns:
            list[int] : The list of indices.
        '''
        return random.sample(range(len(self.test_data)), count)
    
    def _test_step(self, i: int, idx: int) -> None:
        data = self.test_data[idx]
        x, y, y_fake = self.run_inference(data[0], data[1]).values()
        image_paths = self.save_test_images(index=i, x=x, y=y, y_fake=y_fake)

        losses = self.run_losses(y=y, y_fake=y_fake)
        image_path = self.get_current_image_path(index=idx)
        log_entry = {
            'iteration': i+1,
            'test_data_path': image_path,
            'output': {
                'x': image_paths[0],
                'y': image_paths[1],
                'y_fake': image_paths[2]
            },
            'losses': {
                'L1': losses['L1'],
                'SOBEL': losses['SOBEL'],
                'LAPLACIAN': losses['LAPLACIAN']
            } 
        }
        self._write_log_entry(entry=log_entry)

    def run_test(
            self, 
            image_count: int=10, 
            silent: bool=False
        ) -> None:
        '''Test loop function for paired inference.
        
        Args:
            image_count : The number of images from the test dataset to run 
                model inference on.
            silent : If True, iteration count will be printed to the console
                during testing.
        '''
        self.build_test_output_directory()
        self.export_base_test_log()
        self.init_loss_functions()

        indices = self.build_lookup_indices(image_count)
        for i, idx in enumerate(indices):
            if not silent: print(f'Running test iteration {i+1}', flush=True)
            self._test_step(i=i, idx=idx)
