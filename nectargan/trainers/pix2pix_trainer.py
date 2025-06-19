from os import PathLike
from typing import Literal, Any

import torch
from torch import optim

from nectargan.trainers.trainer import Trainer
from nectargan.config.config_manager import ConfigManager

from nectargan.models.unet.model import UnetGenerator
from nectargan.models.patchgan.model import Discriminator
import nectargan.losses.pix2pix_objective as spec

from nectargan.scheduling.scheduler_torch import TorchScheduler
from nectargan.scheduling.data import Schedule

class Pix2pixTrainer(Trainer):
    def __init__(
            self, 
            config: str | PathLike | ConfigManager | None=None,
            loss_subspec: Literal[
                'basic', 
                'basic+vgg', 
                'extended',
                'extended+vgg'
            ] = 'basic',
            log_losses: bool=True
        ) -> None:
        super().__init__(config=config, quicksetup=True, log_losses=log_losses)

        self.extend_loss_spec = 'basic' not in loss_subspec 
        self.vgg_loss_enabled = '+vgg' in loss_subspec 
        self.log_losses = log_losses
           
        self._init_lr_scheduling() # Handle split LR logic 
        self._init_generator()     # Init generator
        self._init_discriminator() # Init discriminator
        self._init_dataloaders()   # Get datasets and create data loaders
        self._init_gradscalers()   # Create gradient scalers

        self._init_losses(loss_subspec) # Init loss functions
        
        if self.config.train.load.continue_train: # Load checkpoint if enabled
            self.load_checkpoint('G', self.gen, self.opt_gen, 
                self.config.train.generator.learning_rate.initial)
            self.load_checkpoint('D', self.disc, self.opt_disc, 
                self.config.train.discriminator.learning_rate.initial)

    ### INITIALIZATION ###

    def _init_lr_scheduling(self) -> None:
        '''If not using separate LR schedules, overrides D LR with G LR.'''
        train = self.config.train
        if not train.separate_lr_schedules:
            train.discriminator.learning_rate = train.generator.learning_rate

    def _init_generator(self) -> None:
        '''Initializes generator with optimizer and lr scheduler.'''
        tg = self.config.train.generator # Get config train data
        self.gen = UnetGenerator( # Init Generator
            input_size=self.config.dataloader.load.crop_size, 
            in_channels=self.config.dataloader.load.input_nc,
            features=tg.features,
            n_downs=tg.n_downs,
            block_type=tg.block_type,
            upconv_type=tg.upsample_type)
        self.gen.to(self.device)  # Cast to current device

        self.opt_gen = self.build_optimizer(
            optimizer=optim.Adam, network=self.gen, 
            lr=tg.learning_rate.initial, beta1=tg.optimizer.beta1)
        
        # Get total epochs and add 1 so the last epoch isn't LR=0.0
        # See: `nectargan.scheduling.data.Schedule`
        total_epochs = (1 + tg.learning_rate.epochs 
                        + tg.learning_rate.epochs_decay)
        self.gen_lr_scheduler = TorchScheduler( # Init LR scheduler
            self.opt_gen, Schedule(
                start_epoch=tg.learning_rate.epochs, 
                end_epoch=total_epochs,
                initial_value=tg.learning_rate.initial, 
                target_value=tg.learning_rate.target))
        
    def _init_discriminator(self) -> None:
        '''Initializes discriminator with optimizer and lr scheduler.'''
        td = self.config.train.discriminator # Get config train data
        self.disc = Discriminator( # Init discriminator
            in_channels=self.config.dataloader.load.input_nc,
            base_channels=td.base_channels,
            n_layers=td.n_layers,
            max_channels=td.max_channels)
        self.disc.to(self.device)  # Cast to current device

        self.opt_disc = self.build_optimizer(
            optimizer=optim.Adam, network=self.disc, 
            lr=td.learning_rate.initial, beta1=td.optimizer.beta1)
        
        # Get total epochs, then add 1 so the last epoch isn't LR=0.0
        # See: `nectargan.scheduling.data.Schedule`
        total_epochs = (1 + td.learning_rate.epochs 
                        + td.learning_rate.epochs_decay)
        self.disc_lr_scheduler = TorchScheduler( # Init LR scheduler
            self.opt_disc, Schedule(
                start_epoch=td.learning_rate.epochs, 
                end_epoch=total_epochs,
                initial_value=td.learning_rate.initial, 
                target_value=td.learning_rate.target)) 
        
    def _init_dataloaders(self) -> None:
        '''Builds dataloaders for training and validation datasets.'''
        self.train_loader = self.build_dataloader('train')
        self.val_loader = self.build_dataloader('val')

    def _init_gradscalers(self) -> None:
        '''Defines gradient scalers for generator and discriminator.

        This function will init the GradScaler device based on the device
        specified in the config file.
        '''
        self.g_scaler = self.d_scaler = torch.amp.GradScaler(self.device)

    def _init_losses(self, loss_subspec: str) -> None:
        '''Initializes model loss functions.

        This function takes `loss_subspec` as input. This defines the subspec 
        to initialize the generator's LossManager with. Valid subspecs for 
        pix2pix are:
            
        'basic': Losses as described in the original paper
            G Loss Funtions: ['G_GAN', 'G_L1']
            D Loss Functions: ['D_real', 'D_fake']
        'basic+vgg': 'basic' + VGG19-based perceptual loss
        'extended': Additional structure loss functions.
            G Loss Funtions: 'basic' + ['G_SOBEL', 'G_LAP']
            D Loss Functions: ['D_real', 'D_fake']
        'extended+vgg': 'extended' + VGG19-based perceptual loss

        This function also shows an example implementation of 
        `LossManager.init_from_spec()`. `spec.pix2pix` is a function that
        registers the LMLoss items for pix2pix model training. It takes as
        arguments a `Config` object (to more easily assign loss weight values) 
        and a str to define the loss subspec, and returns a dict[str, LMLoss]. 
        Passing the funtion reference and it's arguements to `init_from_spec`
        tells the LossManager to run that function, get the return dict, and 
        register all of the losses contained within it.

        For more information on spec functions, please see:

            - `nectargan.losses.pix2pix_objective.pix2pix()`
            - `nectargan.losses.loss_manager.init_from_spec()`

        Args:
            loss_subspec : The subspec to initialize the LossManager with.
        '''
        valid_subspecs = ['basic', 'basic+vgg', 'extended', 'extended+vgg']
        if not loss_subspec in valid_subspecs:
            raise ValueError(
                f'Invalid loss subspec. Valid options are: {valid_subspecs}')
        self.loss_manager.init_from_spec( # Init loss manager
            spec.pix2pix, self.config, loss_subspec)

    ### DATA VISUALIZATION ###

    def update_display(
            self, 
            x: torch.Tensor, 
            y: torch.Tensor, 
            y_fake: torch.Tensor,
            idx: int
        ) -> None:
        '''Function to print losses and update image displays during training.

        Args:
            x : Input image as torch.Tensor.
            y : Ground truth image as torch.Tensor.
            y_fake : Generated fake image as torch.Tensor.
            idx : Batch iteration value from training loop.
        '''
        self.loss_manager.print_losses(self.current_epoch, idx) # Print losses
        
        if self.config.visualizer.visdom.enable:
            self.vis.update_images( # Update x, y_fake, y image grid
                x=x, y=y_fake, z=y, title='real_A | fake_B | real_B', 
                image_size=self.config.visualizer.visdom.image_size)
            
            # Get G and D loss values from LossManager
            losses_G = self.loss_manager.get_loss_values(query=['G'])
            losses_D = self.loss_manager.get_loss_values(query=['D'])

            # Normalize by epoch to get graph step
            graph_step = self.current_epoch + idx / len(self.train_loader) 
            self.vis.update_loss_graphs(graph_step, losses_G, losses_D)

    def print_end_of_epoch(
            self, 
            precision: int=8, 
            capture: bool=False
        ) -> str | None:
        '''Prints information at end of epoch.
        
        The base Trainer class implements this function to print epoch index
        and time taken. This child class adds to that lines to show the changes
        in learning rate between the just completed epoch, and the epoch which
        is about to begin.

        Args:
            precision : Rounding precision of printed learning rates.
        '''
        output = super().print_end_of_epoch(capture=capture)
        if output is None: return ''
        
        # Print prev and new generator learning rate
        gen_lr_step = self.gen_lr_scheduler.get_lr()
        output += (
            f'\nLearning rate (G): {round(gen_lr_step[0], precision)}'
            f' => {round(gen_lr_step[1], precision)}'
        )
        
        # Print prev and new discriminator learning rate
        disc_lr_step = self.disc_lr_scheduler.get_lr()
        output += (
            f'\nLearning rate (D): {round(disc_lr_step[0], precision)}'
            f' => {round(disc_lr_step[1], precision)}'
        )

        if capture: return output
        else: print(output)


    ### DISCRIMINATOR TRAINING ###

    def forward_D(
            self, 
            x: torch.Tensor, 
            y: torch.Tensor, 
            y_fake: torch.Tensor
        ) -> dict[str, torch.Tensor]:
        '''Forward step for discriminator. 
        
        Run each batch to compute discriminator loss, this function first has
        the discriminator make a prediction on a real (x, y) pair, then it has
        it make a prediction on the generators fake output (x, y_fake). Loss is
        computed for each prediction via BCEWithLogits, then the two resulting
        loss tensors are averaged to compute the final discriminator loss for
        the batch.

        Args:
            x : Input image as torch.Tensor.
            y : Ground truth image as torch.Tensor.
            y_fake : Generated fake image as torch.Tensor.
        '''
        with torch.amp.autocast('cuda'):
            D_real = self.disc(x, y)               # Predict on real image
            D_fake = self.disc(x, y_fake.detach()) # Then on fake image

            # Compute loss on real and fake images
            loss_D_real = self.loss_manager.compute_loss_xy(
                'D_real', D_real, torch.ones_like(D_real), self.current_epoch)
            loss_D_fake = self.loss_manager.compute_loss_xy(
                'D_fake', D_fake, torch.zeros_like(D_fake), self.current_epoch)
            
            # Get final discriminator loss by averaging real and fake
            loss_D = (loss_D_real + loss_D_fake) * 0.5 
        
        return { # Return losses
            'loss_D': loss_D,
            'loss_D_real': loss_D_real, 
            'loss_D_fake': loss_D_fake}

    def backward_D(self, loss_D: torch.Tensor) -> None:
        '''Backward step for the discriminator.
        
        This function takes the loss computed during the forward_D step and
        uses it to compute and apply the discriminator network's loss gradients
        for the batch. It first zeroes out the discriminator's loss gradients,
        then it scales loss_D by the GradScalers current scale factor and 
        computes the new loss gradient. This loss gradient is then unscaled in-
        place to check for invalid data (NaN, Inf), then the scaling factor of
        the GradScaler is updated based on the results.

        Args:
            loss_D : Discriminator loss for current batch.
        '''
        self.disc.zero_grad()                  # Clear loss gradients
        self.d_scaler.scale(loss_D).backward() # Scale loss, compute gradient
        self.d_scaler.step(self.opt_disc)      # Optimizer step
        self.d_scaler.update()                 # Update scaling factor

    ### GENERATOR TRAINING ##

    def compute_GAN_loss(
            self, 
            x: torch.Tensor, 
            y_fake: torch.Tensor
        ) -> torch.Tensor:
        '''Computes adversarial loss for the generator network.

        This function first has the discriminator model make a prediction on
        the generated image (or images if batch_size>1), then takes the result
        of that prediction and uses it to compute adversarial loss for the
        generator via BCEWithLogitsLoss, and returns the resulting loss tensor.

        When we invoke the loss function, done here via the LossManager, we
        pass it three arguments. Here, those are:

            loss_name : This this the lookup key for the loss type registered
                with the LossManager. We have BCEWithLogits registered as 
                'G_GAN' in both the 'common' and 'extended' pix2pix specs.
            x : This is the resulting prediction tensor from the discriminator.
            y : This is just a tensor of all 1's. If G was absolutely perfect, 
                D_fake would be a tensor only containing 1's, since G is able 
                to completely fool D every time. The y input tensor here 
                mimicking what D's output would be in that scenario. This is 
                effectively treated as the goal for G when computing
                adversarial loss.

        Args:
            x : Input image as torch.Tensor.
            y_fake : Generated fake image as torch.Tensor.

        Returns:
            torch.Tensor : Generator adversarial loss.
        '''
        D_fake = self.disc(x, y_fake) # Get prediction from discriminator
        return self.loss_manager.compute_loss_xy( # Compute G loss from D_fake
            'G_GAN', D_fake, torch.ones_like(D_fake), self.current_epoch)
    
    def compute_structure_loss(
            self, 
            y: torch.Tensor, 
            y_fake: torch.Tensor
        ) -> tuple[torch.Tensor]:
        '''Computes structural loss (L1, SOBEL, etc.) for the generator model.

        This function computes structural loss for the generator network. If 
        the current loss subspec is 'basic', it will only output the L1 loss 
        result. If it is set to 'extended', however, it will also compute and
        output the results of some additional structure loss functions. Please
        see `nectargan.losses.lm_specs.pix2pix()` for more details.

        Args:
            y : Ground truth image as torch.Tensor.
            y_fake : Generated fake image as torch.Tensor.

        Returns:
            tuple[torch.Tensor] : Generator structural losses.
        '''        
        lm = self.loss_manager # Get loss manager and compute L1 loss
        loss_G_L1 = lm.compute_loss_xy('G_L1', y_fake, y, self.current_epoch)

        loss_G_L2 = torch.zeros_like(loss_G_L1)    # Dummy tensor for MSE
        loss_G_SOBEL = torch.zeros_like(loss_G_L1) # And for sobel
        loss_G_LAP = torch.zeros_like(loss_G_L1)   # And Laplacian
        loss_G_VGG = torch.zeros_like(loss_G_L1)   # And also for VGG
        
        if self.extend_loss_spec: # Extended losses, if enabled 
            loss_G_L2 = lm.compute_loss_xy(
                'G_L2', y_fake, y, self.current_epoch)
            loss_G_SOBEL = lm.compute_loss_xy(
                'G_SOBEL', y_fake, y, self.current_epoch)
            loss_G_LAP = lm.compute_loss_xy(
                'G_LAP', y_fake, y, self.current_epoch)   
        if self.vgg_loss_enabled: # VGG perceptual, if enabled
            loss_G_VGG = lm.compute_loss_xy('G_VGG', y_fake, y)          

        return (loss_G_L1, loss_G_L2, loss_G_SOBEL, loss_G_LAP, loss_G_VGG)
        
    def forward_G(
            self, 
            x: torch.Tensor, 
            y: torch.Tensor, 
            y_fake: torch.Tensor
        ) -> torch.Tensor:
        '''Forward step for generator. 
        
        Run each iteration to compute generator loss, this function first
        computes adversarial loss for the generator, then computes structure
        loss, then it sums them all together to get the final loss of the
        generator network for this batch.

        Args:
            x : Input image as torch.Tensor.
            y : Ground truth image as torch.Tensor.
            y_fake : Generated fake image as torch.Tensor.

        Returns:
            torch.Tensor : Total generator loss for batch.
        '''
        with torch.amp.autocast('cuda'):
            loss_G = self.compute_GAN_loss(x, y_fake)
            structure_losses = self.compute_structure_loss(y, y_fake)
            for loss in structure_losses:
                loss_G += loss # Add structural losses
        return loss_G          # Return generator losses
    
    def backward_G(self, loss_G: torch.Tensor) -> None:
        '''Backward step for the generator. 
        
        Please see `Pix2pixTrainer.backward_D()`.

        One thing to note, though, is that in `backward_D()`, we zero out the
        loss gradients for the discriminator model itself, where here, we
        instead zero the gradients for the optimizer associated with the
        generator model. This does not make a difference for this trainer since
        self.opt_gen manages all of the generator parameters. My general
        understanding of the philosophy behind this idea, though, is that, as
        models become more complex with more parameters to track and manage,
        this more granular approach enforces a sort of clarity in what opt
        owns what parms, etc. Here, though, we have one optimizer per network
        so they effectively do the same thing.

        Args:
            loss_G : Total generator loss for batch.
        '''
        self.opt_gen.zero_grad()               # Clear optimizer loss gradients
        self.g_scaler.scale(loss_G).backward() # Scale loss, compute gradient
        self.g_scaler.step(self.opt_gen)       # Optimizer step
        self.g_scaler.update()                 # Update scaling factor

    def update_schedulers(self) -> None:
        '''Updates model schedulers.
        '''
        self.disc_lr_scheduler.step() # Discriminator LR scheduler
        self.gen_lr_scheduler.step()  # Generator LR scheduler

    ### OVERRIDE METHODS FOR TRAINING CALLBACKS ###

    def on_epoch_start(self, **kwargs: Any) -> None:
        '''Train start override method for Pix2pixTrainer class.

        This method overrides Trainer.on_epoch_start() and is called at the 
        beginning of a training cycle, just before the training loop is 
        started.

        This implementation also demonstrates how to use the additional keyword
        arguments for the Trainer callback functions. In `/scripts/train.py`,
        we are calling `Trainer.train_paired()`, and passing it the current
        epoch, and a dict for the callback_kwargs. This dict contains an entry
        called `on_epoch_start`, same as the callback name, and it's value is
        a dict containing a single bool value, `print_train_start`. 
        
        That is passed through to this function via the training function, so 
        here, we are able to look up that value and, if it is true, we have the 
        script print some training info at the start of each epoch. You can 
        create one dict for each callback (`on_epoch_start`, `train_step`, 
        `on_epoch_end`), and each dict should be added as separate entries to
        a another dict with the key set at the name of the callback you would
        like to pass the kwargs to.

        Example:

            trainer = Pix2pixTrainer()

            for epoch in epochs:
                trainer.train_paired(
                    epoch, 
                    callback_kwargs={
                        'on_epoch_start': { 'var1': 1.0 },
                        'train_step': { 'var2': True, 'var3': [1.0, 2.0] },
                        'on_epoch_end': {'var4': { 'x': 1.0, 'y': 2.0 } }
                    }

        Args:
            **kwargs : Any keyword arguments you would like to pass to the 
                callback during training.
        '''
        if 'print_train_start' in kwargs.items():
            if kwargs['print_train_start']:
                print(f'Beginning epoch: {self.current_epoch}')
                self.loss_manager.print_weights()

    def train_step(
            self, 
            x: torch.Tensor, 
            y: torch.Tensor, 
            idx: int,
            **kwargs: Any
        ) -> None:
        '''Train step callback for pix2pix trainer (see: Trainer.train_paired).
        
        Generator first makes a fake image (y_fake). Then losses are computed 
        for the discriminator based on x, y, and y_fake. Then losses are 
        computed for generator based on discriminator output and additional 
        loss functions. Losses get applied to both networks, then we update
        display if applicable.

        Args:
            x : torch.Tensor of input image.
            y : torch.Tensor of ground image.
            idx : Batch iteration value from training loop.
            **kwargs : Any additional keyword arguments you would like to pass 
                to the callback during training. See 
                `Pix2pixTrainer.on_epoch_start()` for example implementation.
        '''
        with torch.amp.autocast('cuda'): 
            y_fake = self.gen(x)

        # Get discriminator losses, apply gradients
        losses_D = self.forward_D(x, y, y_fake)
        self.backward_D(losses_D['loss_D'])
        
        # Get generator losses, apply gradients
        loss_G = self.forward_G(x, y, y_fake)
        self.backward_G(loss_G)
        
        if idx % self.config.visualizer.visdom.update_frequency == 0:
            self.update_display(x, y, y_fake, idx)
        
    def on_epoch_end(self, **kwargs: Any) -> None:
        '''Train end callback for pix2pix trainer. (see: Trainer.train_paired)

        Args:
            **kwargs : Any keyword arguments you would like to pass to the 
                callback during training. See `Pix2pixTrainer.on_epoch_start()` 
                for example implementation.
        '''
        # Dump stored loss values to log_log.json
        if self.log_losses: # From parent Trainer class
            self.loss_manager.update_loss_log(silent=False)
        
        # Update schedulers after training
        self.update_schedulers()
        
    ### SAVE MODEL/EXAMPLE ###

    def save_checkpoint(
            self, 
            mode: Literal['g', 'd', 'both']='both',
            capture: bool=False
        ) -> str | None:
        '''Saves a .pth.tar checkpoint file named with the current epoch.
        
        Args:
            mode : Defines which checkpoints to save. 
                'g' for only generator,
                'd' for only discriminator, 
                'both' (default) for both.
            capture : If true, this function will return the log strings from
                `Trainer.export_model_weights()` rather than printing them.
        
        Raises:
            ValueError : If input mode is not valid.
        '''    
        output = ''
        match mode.lower():
            case 'g' | 'g': 
                path = self.export_model_weights(
                    self.gen, self.opt_gen, mode.upper())
                output = f'Checkpoint Saved ({mode.upper()}): {path}'
            case 'both':
                path_g = self.export_model_weights(
                    self.gen, self.opt_gen, 'G')
                path_d = self.export_model_weights(
                    self.disc, self.opt_disc, 'D')
                output = (
                    f'Checkpoint Saved (G): {path_g}\n'
                    f'Checkpoint Saved (D): {path_d}')
            case _: 
                message = 'Encountered invalid mode while saving checkpoint.'
                raise ValueError(message)
        if capture: return output
        else: return None

    def save_examples(
            self, 
            silent: bool=False, 
            capture: bool=False
        ) -> str | None:
        '''Evaluates model and saves images to example output directory.'''        
        if not silent:
            output = f'Saving example images: {self.examples_dir.as_posix()}'
            self.save_xyz_examples(network=self.gen, dataloader=self.val_loader)
            if capture: return output
            else: print(output)
        