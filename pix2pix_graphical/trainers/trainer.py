import random
import pathlib
from os import PathLike
from typing import Callable

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image

from pix2pix_graphical.config.config_manager import ConfigManager
from pix2pix_graphical.losses.loss_manager import LossManager
from pix2pix_graphical.visualizer.visdom_visualizer import VisdomVisualizer
from pix2pix_graphical.dataset.pix2pix_dataset import Pix2pixDataset

class Trainer():
    def __init__(
            self, 
            input_config: str | PathLike | ConfigManager | None=None
        ) -> None:
        self.init_config(input_config)
        self.loss_manager = LossManager(self.config) # Init loss manager
        self.init_visualizers() # Init visualizer
        
        self.current_epoch: int | None = None
        self.train_loader: torch.utils.data.DataLoader | None = None
        self.val_loader: torch.utils.data.DataLoader | None = None

    def init_config(self, input_config: str | PathLike | ConfigManager | None):
        '''Handles various input_config types and inits config data accordingly.
        '''
        match input_config:
            case str() | PathLike() | None:
                # ConfigManager handles logic for these cases
                self.config_manager = ConfigManager(input_config)
            case ConfigManager():
                # If input_config is a ConfigManager, we just override self.configmanager
                self.config_manager = input_config
            case _:
                valid_types = [type(str), type(PathLike), type(ConfigManager), None]
                raise ValueError(f'Invalid input config type. Valid types are {valid_types}')
        self.config = self.config_manager.data # Store config data for easier access         

    def init_visualizers(self) -> None:
        vcon = self.config.visualizer # Get visualizer config data
        if vcon.enable_visdom:        # Init Visdom visualizer
            self.vis = VisdomVisualizer(env=vcon.visdom_env_name) 
            self.vis.clear_env()      # Clear Visdom environment    

    def build_output_directory(self) -> None:
        '''Builds an output directory structure for the experiment.'''
        output_root = pathlib.Path(self.config.common.output_directory)
        exp_name = self.config.common.experiment_name       # Get experiment name
        exp_version = self.config.common.experiment_version # Get experiment version
        name = f'{exp_name}_v{exp_version}'                 # Build output dir name
        experiment_dir = pathlib.Path(output_root, name)    # Output dir path
        
        try: # Exists okay is true if we're loading a previously trained model
            experiment_dir.mkdir(parents=False, exist_ok=self.config.load.continue_train)
        except FileExistsError as e: # Otherwise we check to see if auto-increment is on
            if self.config.save.auto_increment_version:
                try:
                    prev_versions = [i.name for i in experiment_dir.parent.iterdir() if exp_name in i.name]
                    prev_versions.sort()
                    new_version = 1 + int(prev_versions[-1].split('v')[-1])
                    experiment_dir = pathlib.Path(output_root, f'{exp_name}_v{new_version}')
                    
                    try: experiment_dir.mkdir(parents=False, exist_ok=False)
                    except Exception: raise
                except Exception as x: raise RuntimeError('Unable to overwrite experiment dir') from x
            else:
                error_message = (
                    f'Unable to save experiment: '
                    f'Experiment directory already exists. '
                    f'Please increment experiment version or enable auto_increment_version.')
                raise RuntimeError(error_message) from e

        examples_dir = pathlib.Path(experiment_dir, 'examples')
        try: examples_dir.mkdir(exist_ok=self.config.load.continue_train)
        except Exception as e:
            raise RuntimeError('Unable to create examples output directory.') from e
        self.experiment_dir, self.examples_dir =  experiment_dir, examples_dir

    def export_config(self) -> None:
        '''Exports a versioned config JSON file to the experiment output directory.
        
        This function just abstracts ConfigManager.export_config for ease of use with
        instances of trainer classes. This is also not strictly necessary to do. The
        Trainer's config data is intialized from the input config file. This just makes
        a copy of it in the experiment output directory for notekeeping purposes.
        '''
        self.config_manager.export_config(self.experiment_dir)

    def build_dataloader(self, loader_type: str):
        '''Initializes a Torch dataloader of the given type from a Pix2pixDataset.'''
        dataset_path = pathlib.Path(self.config.common.dataroot, loader_type).resolve()
        if not dataset_path.exists(): # Make sure data directory exists
            raise Exception(f'Unable to locate dataset at: {dataset_path.as_posix()}')
        dataset = Pix2pixDataset(config=self.config, root_dir=dataset_path)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.config.dataloader.batch_size, 
            shuffle=True, num_workers=self.config.dataloader.num_workers)
    
    def load_checkpoint(
            self, generator: nn.Module, gen_optimizer: optim.Optimizer,
            discriminator: nn.Module, disc_optimizer: optim.Optimizer) -> None:
        '''Loads pre-trained model weights to continue training.
        '''
        load_epoch = self.config.load.load_epoch
        if load_epoch == -1: base_name = f'final'
        else: base_name = f'epoch{load_epoch}'

        output_directory = pathlib.Path(self.config.common.output_directory)
        experiment_directory = pathlib.Path(output_directory, self.config.common.experiment_name)
        
        # Load generator checkpoint
        checkpoint_gen_path = pathlib.Path(experiment_directory, f'{base_name}_netG.pth.tar')
        if not checkpoint_gen_path.exists():
            raise Exception(f'Unable to locate generator checkpoint at: {checkpoint_gen_path.as_posix()}')
        
        # Load discriminator checkpoint
        checkpoint_disc_path = pathlib.Path(experiment_directory, f'{base_name}_netD.pth.tar')
        if not checkpoint_disc_path.exists():
            raise FileNotFoundError(
                f'Unable to locate discriminator checkpoint at: {checkpoint_disc_path.as_posix()}')
        
        models = [
            (checkpoint_gen_path, generator, gen_optimizer), 
            (checkpoint_disc_path, discriminator, disc_optimizer)]
        for path, model, optimizer in models:
            checkpoint = torch.load(path.as_posix(), map_location=self.config.common.device)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            for param_group in optimizer.param_groups:
                param_group['lr'] = self.config.train.learning_rate

    def on_train_start(self, x: torch.Tensor, y: torch.Tensor, idx: int) -> None:
        raise NotImplementedError('on_train_start() is not implemented by the child class.')

    def train_step(self, x: torch.Tensor, y: torch.Tensor, idx: int) -> None:
        raise NotImplementedError('train_step() is not implemented by the child class.')
    
    def on_train_end(self, x: torch.Tensor, y: torch.Tensor, idx: int) -> None:
        raise NotImplementedError('on_train_end() is not implemented by the child class.')

    def train_paired(
            self, epoch:int,
            on_train_start: Callable[[], None] | None=None,
            train_step: Callable[[torch.Tensor, torch.Tensor, int], None] | None=None,
            on_train_end: Callable[[], None] | None=None) -> None:
        '''Paired adversarial training function.
        
        This funtion runs a paired adversarial training loop, the components of which
        can be defined as an override method in the child class, or by passing the 
        function as a callable to the input argument of the same name.

        Args:
            on_train_start : Run once, right before the training loop begins.
            train_step : Run once per batch in the dataset.
            on_train_end : Run once, after all batches have been completed.
        '''
        # self.current_epoch is a sort of human-readable current epoch value
        # Basically just epoch+1 but it also accounts for loaded checkpoints
        if self.config.load.continue_train:
            self.current_epoch = epoch + 1 + self.config.load.load_epoch
        else: self.current_epoch = epoch + 1

        start_fn = on_train_start or self.on_train_start # Init pre-train funtion
        train_fn = train_step or self.train_step         # Init train step funtion
        end_fn = on_train_end or self.on_train_end       # Init post-train funtion
        
        start_fn() # Run pre-train function
        for idx, (x, y) in enumerate(self.train_loader):
            # Get (x, y) of batch[idx] from training dataset
            x, y = x.to(self.config.common.device), y.to(self.config.common.device)
            train_fn(x, y, idx) # Run train step function
        end_fn() # Run post-train function

    def print_end_of_epoch(self, index: int, begin_epoch: float, end_epoch: float) -> None:
        print(f'(End of epoch {index}) Time: {end_epoch - begin_epoch:.2f} seconds', flush=True)

    def export_model_weights(self, mod: nn.Module, opt: optim.Optimizer, net: str) -> None: 
        '''Save a checkpoint for a single network and associated optimizer.
        
        They will be saved to the experiment output directory. File names will be 
        formatted as epoch{current_epoch}_net{model_to_save}.pth.tar.

        Args:
            mod : The network to save.
            opt : The network's optimizer
            net : The name of the network being saved (e.g G for generator).
        
        Raises:
            RuntimeError : If unable to save checkpoint file.
        '''
        checkpoint = {'state_dict': mod.state_dict(), 'optimizer': opt.state_dict(),}
        name = f'epoch{str(self.current_epoch)}_net{net}'
        output_path = pathlib.Path(self.experiment_dir, name)
        print(f'Saving Checkpoint({net}): {output_path.as_posix()}')
        try: torch.save(checkpoint, output_path.as_posix())
        except Exception as e:
            raise RuntimeError(f'Unable to save checkpoint file: {output_path.as_posix()}') from e

    def save_xyz_examples(self, network: nn.Module, dataloader: torch.utils.data.Dataset) -> None:
        '''Evaluates model and saves examples to example output directory.

        This function will switch the generator model to eval mode and select
        a number of random images, defined by [save][num_examples] in the config
        file, from the validation dataset. It will then run the generator on each
        of those input images and export a set of [x, y, y_fake] for each to the
        experiment/examples directory.
        '''        
        network.eval()
        val_data = list(dataloader.dataset)
        indices = random.sample(range(len(val_data)), self.config.save.num_examples)
        for i, idx in enumerate(indices):
            x, y = val_data[idx]
            x = x.unsqueeze(0).to(self.config.common.device)
            y = y.unsqueeze(0).to(self.config.common.device)

            with torch.no_grad():
                y_fake = network(x)
            
            base_name = f'epoch{self.current_epoch}_{str(i+1)}'
            imgs = [(x, 'A_real'), (y, 'B_real'), (y_fake, 'B_fake')]
            for img, id in imgs:
                output_path = pathlib.Path(self.examples_dir, f'{base_name}_{id}.png')
                save_image(img * 0.5 + 0.5, output_path.as_posix())
        network.train()
