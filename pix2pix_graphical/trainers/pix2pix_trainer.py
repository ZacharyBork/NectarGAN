import torch
import torch.nn as nn
import torch.optim as optim
from typing import Literal

from pix2pix_graphical.trainers.trainer import Trainer
from pix2pix_graphical.models.unet_model import UnetGenerator
from pix2pix_graphical.models.patchgan_model import Discriminator

from pix2pix_graphical.utils import scheduler
from pix2pix_graphical.losses.losses import SobelLoss, LaplacianLoss

class Pix2pixTrainer(Trainer):
    def __init__(
            self, 
            config_filepath: str | None=None,
            loss_subspec: Literal['basic', 'extended']='basic'
        ) -> None:
        super().__init__(config_filepath)

        self.extended_loss_spec = loss_subspec != 'basic'

        self.init_generator()     # Init generator
        self.init_discriminator() # Init discriminator
        self.build_dataloaders()  # Get datasets and create data loaders
        self.define_gradscalers() # Create gradient scalers

        self.init_losses(loss_subspec) # Init loss functions
        
        if self.config.load.continue_train: # Load checkpoint if applicable
            self.load_checkpoint(self.gen, self.opt_gen, self.disc, self.opt_disc)

    def build_dataloaders(self) -> None:
        '''Builds dataloaders for training and validation datasets.
        '''
        self.train_loader = self.build_dataloader('train')
        self.val_loader = self.build_dataloader('val')

    def init_generator(self) -> None:
        '''Initializes generator with optimizer and lr scheduler.
        '''
        self.gen = UnetGenerator( # Init Generator
            input_size=self.config.dataloader.crop_size, 
            in_channels=self.config.common.input_nc,
            upconv_type=self.config.generator.upsample_block_type).to(self.config.common.device)
        self.opt_gen = optim.Adam( # Init optimizer
            self.gen.parameters(), 
            lr=self.config.train.learning_rate, 
            betas=(self.config.optimizer.beta1, 0.999))
        self.gen_lr_scheduler = scheduler.LRScheduler( # Init LR scheduler
            self.opt_gen, self.config.train.num_epochs, self.config.train.num_epochs_decay)
        
    def init_discriminator(self) -> None:
        '''Initializes discriminator with optimizer and lr scheduler.
        '''
        self.disc = Discriminator( # Init discriminator
            in_channels=self.config.common.input_nc,
            base_channels=self.config.discriminator.base_channels_d,
            n_layers=self.config.discriminator.n_layers_d,
            max_channels=self.config.discriminator.max_channels_d).to(self.config.common.device)
        self.opt_disc = optim.Adam( # Init optimizer
            self.disc.parameters(), 
            lr=self.config.train.learning_rate, 
            betas=(self.config.optimizer.beta1, 0.999))
        self.disc_lr_scheduler = scheduler.LRScheduler( # Init LR scheduler
            self.opt_disc, self.config.train.num_epochs, self.config.train.num_epochs_decay) 
        
    def define_gradscalers(self) -> None:
        '''Defines gradient scalers for generator and discriminator
        based on current selected device.
        '''
        if self.config.common.device == 'cpu':
            self.g_scaler = self.d_scaler = torch.amp.GradScaler('cpu')
        else: self.g_scaler = self.d_scaler = torch.amp.GradScaler('cuda')

    def init_losses(self, loss_subspec: Literal['basic', 'extended']) -> None:
        '''Initializes model loss functions.

        This function takes `loss_subspec: Literal['basic', 'extended']`
        as input. This defines the subspec to initialize the generator's
        LossManager with. Valid subspecs for pix2pix are:
            
        'basic': Losses as described in the original paper
            G Loss Funtions: ['G_GAN', 'G_L1']
            D Loss Functions: ['D_real', 'D_fake']
        'extended': Additional structure loss functions.
            G Loss Funtions: 'basic' + ['G_SOBEL', 'G_LAP']
            D Loss Functions: ['D_real', 'D_fake']

        Args:
            loss_subspec : The subspec to initialize the LossManager with.
        '''
        # Per LossManager.register_pix2pix_losses(), default
        # loss subspecs for pix2pix are ['basic', 'extended']
        pix2pix_subspecs = ['basic', 'extended']
        if not loss_subspec in pix2pix_subspecs:
            raise ValueError(
                f'Invalid loss subspec. Valid options are: {pix2pix_subspecs}')
        self.loss_manager.init_from_spec(spec='pix2pix', sub_spec=loss_subspec)

    def update_display(self, x: torch.Tensor, y: torch.Tensor, y_fake: torch.Tensor,idx: int) -> None:
        '''Function to print losses and update image displays during training.

        Args:
            x : Input image as torch.Tensor.
            y : Ground truth image as torch.Tensor.
            y_fake : Generated fake image as torch.Tensor.
            idx : Batch iteration value from training loop.
        '''
        self.loss_manager.print_losses(self.current_epoch, idx) # Print current loss values
        
        if self.config.visualizer.enable_visdom:
            self.vis.update_images( # Update x, y_fake, y image grid
                x=x, y=y_fake, z=y, title='real_A | fake_B | real_B', 
                image_size=self.config.visualizer.visdom_image_size)
            
            losses_G = self.loss_manager.get_loss_values('G')
            losses_D = self.loss_manager.get_loss_values('D')

            num_batches = len(self.train_loader) # Get batch count per epoch
            graph_step = self.current_epoch + idx / num_batches # Epoch normalized graph step
            self.vis.update_loss_graphs(graph_step, losses_G, losses_D) # Update loss graphs

    def print_end_of_epoch(self, begin_epoch: float, end_epoch: float) -> None:
        '''Prints information at end of epoch.
        
        The base Trainer class implements this function to epoch index and time
        taken. This child class adds to that lines to show the changes in
        learning rate between the just completed epoch, and the epoch which is
        about to being.

        Args:
            begin_epoch : Epoch start time.
            end_epoch : Epoch end time.
        '''
        super().print_end_of_epoch(self.current_epoch, begin_epoch, end_epoch)
        gen_lr_step = self.gen_lr_scheduler.get_lr()
        print(f'Learning rate (G): {gen_lr_step[0]} => {gen_lr_step[1]}')
        disc_lr_step = self.disc_lr_scheduler.get_lr()
        print(f'Learning rate (D): {disc_lr_step[0]} => {disc_lr_step[1]}')

    def forward_D(
            self, 
            x: torch.Tensor, 
            y: torch.Tensor, 
            y_fake: torch.Tensor
        ) -> dict[str, torch.Tensor]:
        '''Forward step for discriminator. 
        
        Run each batch to compute discriminator loss.

        Args:
            x : Input image as torch.Tensor.
            y : Ground truth image as torch.Tensor.
            y_fake : Generated fake image as torch.Tensor.
        '''
        with torch.amp.autocast('cuda'):
            D_real = self.disc(x, y)               # Make prediction on real image
            D_fake = self.disc(x, y_fake.detach()) # Then on fake image
            loss_D_real = self.loss_manager.compute_loss_xy('D_real', D_real, torch.ones_like(D_real))
            loss_D_fake = self.loss_manager.compute_loss_xy('D_fake', D_fake, torch.zeros_like(D_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5 # Normalized loss gradient
        return { # Return losses
            'loss_D': loss_D,
            'loss_D_real': loss_D_real, 
            'loss_D_fake': loss_D_fake
        }

    def backward_D(self, loss_D: torch.Tensor) -> None:
        '''Backward step for the discriminator.
        
        Computes and applies gradients from discriminator loss and updates optimizer 
        scaling accordingly.

        Args:
            loss_D : Discriminator loss for current batch.
        '''
        self.disc.zero_grad()                  # Clear loss gradients
        self.d_scaler.scale(loss_D).backward() # Scale loss and compute gradient
        self.d_scaler.step(self.opt_disc)      # Optimizer step
        self.d_scaler.update()                 # Update scaling factor
        
    def compute_GAN_loss(self, x: torch.Tensor, y_fake: torch.Tensor) -> torch.Tensor:
        '''Computes adversarial loss (BCEWithLogits) for the generator model.

        Args:
            x : Input image as torch.Tensor.
            y_fake : Generated fake image as torch.Tensor

        Returns:
            torch.Tensor : Generator adversarial loss.
        '''
        D_fake = self.disc(x, y_fake) # Get prediction from discriminator
        loss = self.loss_manager.compute_loss_xy('G_GAN', D_fake, torch.ones_like(D_fake))
        return loss * self.config.loss.lambda_gan
    
    def compute_structure_loss(
            self, 
            y: torch.Tensor, 
            y_fake: torch.Tensor
        ) -> tuple[torch.Tensor]:
        '''Computes structural loss (L1, SOBEL, etc.) for the generator model.

        Args:
            y : Ground truth image as torch.Tensor.
            y_fake : Generated fake image as torch.Tensor.

        Returns:
            tuple[torch.Tensor] : Generator structural losses.
        '''        
        loss_G_L1 = self.loss_manager.compute_loss_xy('G_L1', y_fake, y)
        loss_G_SOBEL = torch.zeros_like(loss_G_L1)
        loss_G_LAP = torch.zeros_like(loss_G_L1)
        
        if self.extended_loss_spec:
            loss_G_SOBEL = self.loss_manager.compute_loss_xy('G_SOBEL', y_fake, y)
            loss_G_LAP = self.loss_manager.compute_loss_xy('G_LAP', y_fake, y)            

        return (
            loss_G_L1 * self.config.loss.lambda_l1, 
            loss_G_SOBEL * self.config.loss.lambda_sobel, 
            loss_G_LAP * self.config.loss.lambda_laplacian)
        
    def forward_G(
            self, 
            x: torch.Tensor, 
            y: torch.Tensor, 
            y_fake: torch.Tensor
        ) -> torch.Tensor:
        '''Forward step for generator. 
        
        Run each iteration to compute generator loss.

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
        
        Computes and applies gradients from generator loss and updates optimizer 
        scaling accordingly.

        Args:
            loss_G : Total generator loss for batch.
        '''
        self.opt_gen.zero_grad() # Clear generator optimizer loss gradients
        self.g_scaler.scale(loss_G).backward() # Scale loss and compute gradient
        self.g_scaler.step(self.opt_gen) # Optimizer step
        self.g_scaler.update() # Update scaling factor

    def update_schedulers(self) -> None:
        '''Updates model schedulers.
        '''
        self.disc_lr_scheduler.step() # Discriminator LR scheduler
        self.gen_lr_scheduler.step()  # Generator LR scheduler

    def on_train_start(self) -> None:
        '''Train start callback for pix2pix trainer. (see: Trainer.train_paired)
        '''
        pass

    def train_step(self, x: torch.Tensor, y: torch.Tensor, idx: int) -> None:
        '''Train step callback for pix2pix trainer (see: Trainer.train_paired).
        
        Generator first makes a fake image (y_fake). Then losses are computed for 
        the discriminator based on x, y, and y_fake. Then losses are computed for 
        generator based on discriminator output and additional loss functions. 
        Losses get applied to both networks, then we update display if applicable.

        Args:
            x : torch.Tensor of input image.
            y : torch.Tensor of ground image.
            idx : Batch iteration value from training loop.
        '''
        with torch.amp.autocast('cuda'): 
            y_fake = self.gen(x)

        # Get discriminator losses, apply gradients
        losses_D = self.forward_D(x, y, y_fake)
        self.backward_D(losses_D['loss_D'])
        
        # Get generator losses, apply gradients
        loss_G = self.forward_G(x, y, y_fake)
        self.backward_G(loss_G)
        
        if idx % self.config.visualizer.update_frequency == 0:
            self.update_display(x, y, y_fake, idx)
        
    def on_train_end(self) -> None:
        '''Train end callback for pix2pix trainer. (see: Trainer.train_paired)
        '''
        # Update schedulers after training
        self.update_schedulers()
        
    def save_checkpoint(self, mode: Literal['g', 'd', 'both'] = 'both') -> None:
        '''Saves a .pth.tar checkpoint file named with the current epoch.
        
        Args:
            mode : Defines which checkpoints to save. 
                'g' for only generator,
                'd' for only discriminator, 
                'both' (default) for both.
        
        Raises:
            ValueError : If input mode is not valid.
        '''        
        match mode:
            case 'g': self.export_model_weights(self.gen, self.opt_gen, 'G')
            case 'd': self.export_model_weights(self.disc, self.opt_disc, 'D')
            case 'both':
                self.export_model_weights(self.gen, self.opt_gen, 'G')
                self.export_model_weights(self.disc, self.opt_disc, 'D')
            case _: raise ValueError('Encountered invalid mode while saving checkpoint.')

    def save_examples(self) -> None:
        '''Evaluates model and saves examples to example output directory.
        '''        
        self.save_xyz_examples(network=self.gen, dataloader=self.val_loader)