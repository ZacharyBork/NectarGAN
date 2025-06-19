import sys
import random
import pathlib
import time
from os import PathLike
from typing import Callable, Any, Literal

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image

from nectargan.config.config_manager import ConfigManager
from nectargan.losses.loss_manager import LossManager
from nectargan.visualizer.visdom.visualizer import VisdomVisualizer
from nectargan.dataset.paired_dataset import PairedDataset

class Trainer():
    def __init__(
            self, 
            config: str|PathLike|ConfigManager|dict[str, Any]|None=None,
            quicksetup: bool=True,
            log_losses: bool=True
        ) -> None:
        '''Init function for the base Trainer class.

        Args:
            config: Something representing a training config, either a str or
                os.Pathlike object pointing to a config JSON, or a Python dict
                representing the data from a config JSON, or a pre-defined
                ConfigManager instance, or None (default) to init from the
                default config located at (/root/config/default.json).
            quicksetup : If enabled (default), the `Trainer` will run a 
                quicksetup during its init function. This will automatically 
                create an output directory for the experiment from the settings 
                in the config and output a copy of the config JSON to that 
                directory, create and assign a `LossManager` instance to 
                manager manage losses during training, and initialize Visdom to 
                visualize results.
            log_losses : If enabled (default) for a given `Trainer` instance, 
                losses run within the context of the `Trainer` will be cached
                and periodically dumped to the loss log JSON.
        '''
        # These are either set by child classes, or passed by training script
        self.log_losses = log_losses
        self.current_epoch: int | None = None
        self.last_epoch_time: float = 0.0
        self.train_loader: torch.utils.data.DataLoader | None = None
        self.val_loader: torch.utils.data.DataLoader | None = None

        self.init_config(config)                # Init config
        self.device = self.config.common.device # Store device for easy lookup
        if quicksetup: self.quicksetup()        # Do quicksetup if applicable

    ### TRAINER QUICKSETUP ###

    def quicksetup(self) -> None:
        '''Function to quickly perform all init steps for Trainer instance.
        '''
        self.build_output_directory()  # Build experiment output directory    
        self.init_loss_manager()       # Init LossManager
        self.export_config()           # Export config file
        if self.config.visualizer.visdom.enable:
            self.init_visdom()         # Init visualizer

    ### INITIALIZATION HELPERS ###

    def init_config(
            self, 
            config: str | PathLike | ConfigManager | dict[str, Any] | None
        ) -> None:
        '''Handles input config types and inits config data accordingly.

        Args:
            config : The config to init from. This can be a Pathlike object to 
                pointing to a config JSON, a dict object representing a config 
                JSON, a ConfigManager instance, or None to init from the 
                default config (/root/config/default.json)
        '''
        match config:
            case str() | PathLike() | dict() | None:
                # ConfigManager handles logic for: str() | PathLike() | None
                self.config_manager = ConfigManager(config)
            case ConfigManager():            # If config is a ConfigManager
                self.config_manager = config # Override self.config_manager
            case _:
                x = [type(str), type(PathLike), type(ConfigManager), None]
                raise ValueError(f'Invalid config type. Valid types are {x}')
        self.config = self.config_manager.data # Store values for easier access 

    def build_output_directory(self) -> None:
        '''Builds an output directory structure for the experiment.
        
        Raises:
            FileExistsError : If config.save.auto_increment_version=False and
                experiment directory already exists.
            RuntimeError : If unable to increment experiment directory.
            RuntimeError : If unable to create examples output directory.
        '''
        output_root = pathlib.Path(self.config.common.output_directory)
        exp_name = self.config.common.experiment_name       # Experiment name
        exp_version = self.config.common.experiment_version # Version
        name = f'{exp_name}_v{exp_version}'                 # Output dir name
        experiment_dir = pathlib.Path(output_root, name)    # Output dir path
        
        try: 
            experiment_dir.mkdir( # Exists okay=True if loading a checkpoint
                parents=False, exist_ok=self.config.train.load.continue_train)
        except FileExistsError as e: 
            if self.config.save.auto_increment_version:
                try: # If experiment dir exists and auto_increment enabled
                    prev_versions = [ # Try to increment experiment dir
                        i.name for i in experiment_dir.parent.iterdir() 
                        if exp_name in i.name]
                    prev_versions.sort()
                    new_version = 1 + int(prev_versions[-1].split('v')[-1])
                    new_name = f'{exp_name}_v{new_version}'
                    experiment_dir = pathlib.Path(output_root, new_name)
                    experiment_dir.mkdir(parents=False, exist_ok=False)
                except Exception as x: 
                    message = 'Unable to increment experiment dir.'
                    raise RuntimeError(message) from x
            else: # But if auto-increment is disabled, raise error instead
                error_message = (
                    f'Unable to save experiment: '
                    f'Experiment directory already exists. '
                    f'Please increment experiment version or '
                    f'enable auto_increment_version.')
                raise FileExistsError(error_message) from e

        examples_dir = pathlib.Path(experiment_dir, 'examples')
        try: examples_dir.mkdir(exist_ok=self.config.train.load.continue_train)
        except Exception as e:
            message = 'Unable to create examples output directory.'
            raise RuntimeError(message) from e
        self.experiment_dir, self.examples_dir =  experiment_dir, examples_dir

    def export_config(self) -> None:
        '''Exports a versioned config file to the experiment output directory.
        
        This function just abstracts `ConfigManager.export_config` for ease of 
        use with instances of Trainer classes. This is also not strictly 
        necessary to do. The Trainer's config data is intialized from the input 
        config file. This just makes a copy of it in the experiment output 
        directory for notekeeping purposes.
        '''
        self.config_manager.export_config(self.experiment_dir)

    def init_visdom(self) -> None:
        '''Initializes visdom visualization.
        '''
        vcon = self.config.visualizer # Get visualizer config data
        if vcon.visdom.enable:        # Init Visdom visualizer
            self.vis = VisdomVisualizer(
                env=vcon.visdom.env_name,
                port=vcon.visdom.port) 
            self.vis.clear_env()      # Clear Visdom environment  

    def load_checkpoint(
            self,
            net_type: Literal['G', 'g', 'D', 'd'],
            network: nn.Module,
            optimizer: optim.Optimizer | None=None,
            learning_rate: float | None=None
        ) -> None:
        '''Loads pre-trained model weights to continue training.

        Args:
            net_type : Type of network to load weights for (i.e. 'G', 'D')
            network : The network to load the weights into.
            optimizer : The network's associated optimizer, or none to only
                load network checkpoint.
            learning_rate : The learning rate to load the network with.
        '''
        load_epoch = self.config.train.load.load_epoch
        base_name = f'epoch{load_epoch}'

        # Load checkpoint
        checkpoint_path = pathlib.Path(
            self.experiment_dir, 
            f'{base_name}_net{net_type.upper()}.pth.tar')
        if not checkpoint_path.exists():
            message = 'Unable to locate checkpoint at: {}'
            raise FileNotFoundError(message.format(checkpoint_path.as_posix()))
        
        checkpoint = torch.load(
            checkpoint_path.as_posix(), map_location=self.device)
        network.load_state_dict(checkpoint['state_dict'])
        if not optimizer is None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

    ### TRAINING COMPONENT BUILDERS ###

    def init_loss_manager(self, override: LossManager | None=None) -> None:
        '''Initializes a loss manager for the Trainer instance.

        Args:
            override : A pre-initialized LossManager to assign to the Trainer, 
                or None (default) to create and assign a new LossManager.

        Raises:
            AssertationError : If override is not LossManager or None.
        '''
        if override is None:
            self.loss_manager = LossManager( # Init loss manager
                self.config, 
                self.experiment_dir, 
                enable_logging=self.log_losses) 
            if self.log_losses: # If loss_logging=True, save initial loss log
                self.loss_manager.export_base_log() 
        else: 
            assert isinstance(override, LossManager)
            self.loss_manager = override

    def build_dataloader(
            self, 
            loader_type: str
        ) -> torch.utils.data.DataLoader:
        '''Initializes a dataloader of the given type from a PairedDataset.

        This function will grab the dataroot path from the config and use it to
        first create a PairedDataset instance of the given loader type, then 
        use that dataset to create and return a Torch Dataloader.

        Args:
            loader_type : Type of loader to create (e.g. 'train', 'val')

        Returns:
            torch.utils.data.DataLoader : Dataloader created from the dataset.
        '''
        dataset_path = pathlib.Path(  # Get global sys path to data
            self.config.dataloader.dataroot, loader_type).resolve()
        if not dataset_path.exists(): # Make sure data directory exists
            message = f'Unable to locate dataset at: {dataset_path.as_posix()}'
            raise FileNotFoundError(message)
        dataset = PairedDataset(config=self.config, root_dir=dataset_path)
        return torch.utils.data.DataLoader( # Build dataloader from dataset
            dataset, batch_size=self.config.dataloader.batch_size, 
            shuffle=True, num_workers=self.config.dataloader.num_workers)
    
    def build_optimizer(
            self, 
            network: nn.Module,
            optimizer: torch.optim,
            lr: float,
            beta1: float
        ) -> torch.optim:
        '''Constructs and returns an optimizer for the given network.
        
        Args:
            network: Network to build the optimizer for.
            optimizer: Type of optimizer to build.
            learning_rate: Learning rate for the new optimizer.
            beta1: Momentum term for optimizer.
        '''
        return optimizer(network.parameters(), lr=lr, betas=(beta1, 0.999))

    ### TRAINING CALLBACK ###

    def on_epoch_start(self, **kwargs: Any) -> None:
        '''Train start callback function.
        
        This function is meant to overridden by the child class. It is called
        at the beginning of a training cycle, just before the training loop is
        started. It is fairly open ended and can be populated with whatever you
        would like. Print statements, value updates, etc.

        Args:
            **kwargs : Any keyword arguments you would like to pass to the 
                callback during training. See `Pix2pixTrainer.on_epoch_start()` 
                for example implementation.

        Raises:
            NotImplementedError : If training is run and this callback is not 
                implemented in the child class that is initiating the training.
        '''
        message = 'on_epoch_start() is not implemented by the child class.'
        raise NotImplementedError(message)

    def train_step(
            self,  
            x: torch.Tensor, 
            y: torch.Tensor, 
            idx: int,
            **kwargs: Any
        ) -> None:
        '''Train step callback function.
        
        This function is meant to overridden by the child class. It is inside
        of the training loop, once per batch, and is used as a core training
        function for the child Trainer. If you are writing a custom Trainer,
        this callback is where you would run your forward and backward steps
        for both the generator, and the discriminator. See `train_step()`
        implementation in Pix2pixTrainer for more info.

        Args:
            x : First input tensor, passed to function via `train_paired()`.
            y : Second input tensor, passed to function via `train_paired()`.
            idx : Batch iter value, passed to function via `train_paired()`.
            **kwargs : Any additional keyword arguments you would like to pass 
                to the callback during training. See 
                `Pix2pixTrainer.on_epoch_start()` for example implementation.

        Raises:
            NotImplementedError : If training is run and this callback is not 
                implemented in the child class that is initiating the training.
        '''
        message = 'train_step() is not implemented by the child class.'
        raise NotImplementedError(message)
    
    def on_epoch_end(self, **kwargs: Any) -> None:
        '''Train end callback function.
        
        This function is meant to overridden by the child class. It is called
        at the end of a training cycle, just after the training loop has
        finished all batches for the epoch. It is fairly open ended and can be 
        populated with whatever you would like. Print statements, value 
        updates, etc. The Pix2pixTrainer class uses this to dump loss history 
        to the logs and updated the learning rate schedulers.

        Args:
            **kwargs : Any keyword arguments you would like to pass to the 
                callback during training. See `Pix2pixTrainer.on_epoch_start()` 
                for example implementation.

        Raises:
            NotImplementedError : If training is run and this callback is not 
                implemented in the child class that is initiating the training.
        '''
        message = 'on_epoch_end() is not implemented by the child class.'
        raise NotImplementedError(message)

    ### TRAINING LOOP ###

    def _train_paired_core(
            self, 
            train_step_fn: Callable[[torch.Tensor, torch.Tensor, int], None],
            train_step_kwargs: dict[str, Any]
        ) -> None:
        '''Paired adversarial training loop.
        
        Args:
            train_step_fn : Train step function, Run once per batch.
            train_step_kwargs : Optional keyword args for train step function.
        '''
        for idx, (x, y) in enumerate(self.train_loader):
            # Loop through (x, y) of batch[idx] from training dataset
            x, y = x.to(self.device), y.to(self.device)
            train_step_fn(x, y, idx, **train_step_kwargs)

    def train_paired(
            self, 
            epoch:int,
            on_epoch_start: Callable[[], None] | None=None,
            train_step: Callable[[torch.Tensor, torch.Tensor, int], None] | 
            None=None,
            on_epoch_end: Callable[[], None] | None=None,
            multithreaded: bool=True,
            callback_kwargs: dict[str, dict[str, Any]] = {}
        ) -> None:
        '''Paired adversarial training function.
        
        This funtion runs a paired adversarial training loop, the components of 
        which can be defined as an override method in the child class, or by 
        passing the function as a callable to the input argument of the same 
        name.

        Args:
            epoch : The iteration value of the training loop at the time this
                function is called. Used to set the `Trainer`'s `current_epoch`
                value (i.e. `Trainer.current_epoch == 1+epoch`)
            on_epoch_start : Run once, right before the training loop begins.
            train_step : Run once per batch in the dataset.
            on_epoch_end : Run once, after all batches have been completed.
            multithreaded : If True (default), this function will start a new 
                thread to update the Visdom visualizers. If False, it will
                update them in the same thread used for training.
            callback_kwargs : A dict[str, dict[str, Any]] with any keyword
                arguments you would like to pass to the callback functions
                during training. kwargs are parsed internally and passed to
                their repective callbacks. The dict should be formatted as 
                follows:

                    callback_kwargs = {
                        'on_epoch_start': { 'var1': 1.0 },
                        'train_step': { 'var2': True, 'var3': [1.0, 2.0] },
                        'on_epoch_end': {'var4': { 'x': 1.0, 'y': 2.0 } }
                    }

                See `Pix2pixTrainer.on_epoch_start()` for example 
                implementation.
        '''
        # self.current_epoch is a sort of human-readable current epoch value
        # Basically just epoch+1 but it also accounts for loaded checkpoints
        if self.config.train.load.continue_train:
            self.current_epoch = 1 + epoch + self.config.train.load.load_epoch
        else: self.current_epoch = epoch + 1

        start_fn = on_epoch_start or self.on_epoch_start # Init pre-train fn
        train_fn = train_step or self.train_step         # Init train step fn
        end_fn = on_epoch_end or self.on_epoch_end       # Init post-train fn

        start_time = time.perf_counter() # Start epoch time
        
        # Run pre-train function
        start_fn(**callback_kwargs.get('on_epoch_start', {})) 
        if multithreaded and self.config.visualizer.visdom.enable: 
            try:
                self.vis.start_thread()
                self._train_paired_core(
                    train_fn, callback_kwargs.get('train_step', {}))
            except KeyboardInterrupt:
                sys.exit('Interrupt Recieved: Stopping training...')
            finally: self.vis.stop_thread()
        else: self._train_paired_core(
            train_fn, callback_kwargs.get('train_step', {}))
        
        # Run post-train function
        end_fn(**callback_kwargs.get('on_epoch_end', {})) 

        end_time = time.perf_counter() # End epoch time
        self.last_epoch_time = end_time-start_time

    ### CONSOLE LOGGING ###

    def print_end_of_epoch(self, capture: bool=False) -> str | None:
        '''Print function for epoch end.
        
        When called from a training script, usually at the end of each epoch,
        the function prints a line to the console tagged with the index of the
        epoch that just concluded, and the time it took to complete the epoch.
        '''
        message = (
            f'(End of epoch {self.current_epoch}) '
            f'Time: {self.last_epoch_time:.2f} seconds')
        if not capture: print(message, flush=True)
        else: return message

    ### MODEL/EXAMPLE SAVE ###

    def export_model_weights(
            self,
            mod: nn.Module, 
            opt: optim.Optimizer, 
            net: str,
        ) -> str | None: 
        '''Save a checkpoint for a single network and associated optimizer.
        
        They will be saved to the experiment output directory. File names will 
        be formatted as epoch{current_epoch}_net{model_to_save}.pth.tar.

        Args:
            mod : The network to save.
            opt : The network's optimizer
            net : The name of the network being saved (e.g G for generator).

        Returns:
            str : System path of the exported checkpoint file.
        
        Raises:
            RuntimeError : If unable to save checkpoint file.
        '''
        checkpoint = {
            'state_dict': mod.state_dict(), 
            'optimizer': opt.state_dict()}
        name = f'epoch{str(self.current_epoch)}_net{net}.pth.tar'
        output_path = pathlib.Path(self.experiment_dir, name)
        try: torch.save(checkpoint, output_path.as_posix())
        except Exception as e:
            message = 'Unable to save checkpoint file: {}'
            raise RuntimeError(message.format(output_path.as_posix())) from e
        return output_path.resolve().as_posix()

    def save_xyz_examples(
            self, 
            network: nn.Module, 
            dataloader: torch.utils.data.Dataset
        ) -> None:
        '''Evaluates model and saves examples to example output directory.

        This function will switch the generator model to eval mode and select
        a number of random images, defined by save.num_examples in the config
        file, from the validation dataset. It will then run the generator on 
        each of those input images and export a set of [x, y, y_fake] for each 
        to the experiment/examples directory.

        Args:
            network : The network to perform the example inference.
            dataloader : The dataloader containing the validation set to eval
                the model on.
        '''               
        network.eval()
        val_data = list(dataloader.dataset)
        indices = random.sample(
            range(len(val_data)), 
            self.config.save.num_examples)
        for i, idx in enumerate(indices):
            x, y = val_data[idx]
            x = x.unsqueeze(0).to(self.device)
            y = y.unsqueeze(0).to(self.device)

            with torch.no_grad():
                y_fake = network(x)
            
            base_name = f'epoch{self.current_epoch}_{str(i+1)}'
            imgs = [(x, 'A_real'), (y, 'B_real'), (y_fake, 'B_fake')]
            for img, id in imgs:
                filename = f'{base_name}_{id}.png'
                output_path = pathlib.Path(self.examples_dir, filename)
                save_image(img * 0.5 + 0.5, output_path.as_posix())
        network.train()
