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
    def __init__(self, config_filepath = None):
        super().__init__(config_filepath)

        self.init_generator()     # Init generator
        self.init_discriminator() # Init discriminator
        self.build_dataloaders()  # Get datasets and create data loaders
        self.define_gradscalers() # Create gradient scalers
        self.init_losses()        # Init loss functions
        
        if self.config.load.continue_train: # Load checkpoint if applicable
            self.load_checkpoint(self.gen, self.opt_gen, self.disc, self.opt_disc)

    def build_dataloaders(self) -> None:
        '''Builds dataloaders for training and validation datasets.'''
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

    def init_losses(self) -> None:
        '''Initializes model loss functions.'''
        self.BCE = nn.BCEWithLogitsLoss().to(self.config.common.device)
        self.L1_LOSS = nn.L1Loss().to(self.config.common.device)
        self.SOBEL_LOSS = SobelLoss().to(self.config.common.device)
        self.LAP_LOSS = LaplacianLoss().to(self.config.common.device)

    def update_display(
            self, x: torch.Tensor, y: torch.Tensor, y_fake: torch.Tensor,
            idx: int, losses_G: dict[str, torch.Tensor], losses_D: dict[str, torch.Tensor]) -> None:
        '''Function to update loss and image displays during training.'''
        losses = {
            'G_GAN': round(losses_G['loss_G_GAN'].item(), 2),
            'G_L1': round(losses_G['loss_G_L1'].item(), 2),
            'G_SOBEL': round(losses_G['loss_G_SOBEL'].item(), 2),
            'G_LAP': round(losses_G['loss_G_LAP'].item(), 2),
            'D_real': round(losses_D['loss_D_real'].item(), 2),
            'D_fake': round(losses_D['loss_D_fake'].item(), 2)
        }
        self.print_losses(idx, losses) # Print current loss values
        
        if self.config.visualizer.enable_visdom:
            self.vis.update_images( # Update x, y_fake, y image grid
                x=x, y=y_fake, z=y, title='real_A | fake_B | real_B', 
                image_size=self.config.visualizer.visdom_image_size)

            num_batches = len(self.train_loader) # Get batch count per epoch
            graph_step = self.current_epoch + idx / num_batches # Epoch normalized graph step
            self.vis.update_loss_graphs(losses, graph_step) # Update loss graphs

    def print_end_of_epoch(self, begin_epoch: float, end_epoch: float) -> None:
        super().print_end_of_epoch(self.current_epoch, begin_epoch, end_epoch)
        gen_lr_step = self.gen_lr_scheduler.get_lr()
        print(f'Learning rate (G): {gen_lr_step[0]} => {gen_lr_step[1]}')
        disc_lr_step = self.disc_lr_scheduler.get_lr()
        print(f'Learning rate (D): {disc_lr_step[0]} => {disc_lr_step[1]}')

    def forward_D(self, x: torch.Tensor, y: torch.Tensor, y_fake: torch.Tensor) -> dict[str, torch.Tensor]:
        '''Forward step for discriminator. Run each iteration to compute
        discriminator loss.
        '''
        with torch.amp.autocast('cuda'):
            D_real = self.disc(x, y) # Make prediction on real image
            D_fake = self.disc(x, y_fake.detach()) # Then on fake image
            loss_D_real = self.BCE(D_real, torch.ones_like(D_real)) # Get real image loss
            loss_D_fake = self.BCE(D_fake, torch.zeros_like(D_fake)) # Get fake image loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5 # Normalized loss gradient
        return { # Return losses
            'loss_D': loss_D,
            'loss_D_real': loss_D_real, 
            'loss_D_fake': loss_D_fake
        }

    def backward_D(self, loss_D: torch.Tensor) -> None:
        '''Backward step for the discriminator. Computes and applies gradients from
        discriminator loss and updates optimizer scaling accordingly.
        '''
        self.disc.zero_grad() # Clear loss gradients
        self.d_scaler.scale(loss_D).backward() # Scale loss and compute gradient
        self.d_scaler.step(self.opt_disc) # Optimizer step
        self.d_scaler.update() # Update scaling factor
        
    def compute_GAN_loss(self, x: torch.Tensor, y_fake: torch.Tensor) -> torch.Tensor:
        D_fake = self.disc(x, y_fake) # Get prediction from discriminator
        return self.BCE(D_fake, torch.ones_like(D_fake))
    
    def compute_structure_loss(self, x: torch.Tensor, y: torch.Tensor, y_fake: torch.Tensor) -> tuple[torch.Tensor]:
        loss_G_L1 = self.L1_LOSS(y_fake, y)
        loss_G_SOBEL = self.SOBEL_LOSS(y_fake, y) if self.config.loss.do_sobel_loss else torch.zeros_like(loss_G_L1)
        loss_G_LAP = self.LAP_LOSS(y_fake, y) if self.config.loss.do_laplacian_loss else torch.zeros_like(loss_G_L1)
        return (loss_G_L1, loss_G_SOBEL, loss_G_LAP)
        
    def forward_G(self, x: torch.Tensor, y: torch.Tensor, y_fake: torch.Tensor) -> dict[str, torch.Tensor]:
        '''Forward step for generator. Run each iteration to compute generator loss.
        '''
        with torch.amp.autocast('cuda'):
            loss_G_GAN = self.compute_GAN_loss(x, y_fake)
            loss_G_L1, loss_G_SOBEL, loss_G_LAP = self.compute_structure_loss(x, y, y_fake)
            loss_G = ( # Calculate final generator loss
                loss_G_GAN * self.config.loss.lambda_gan + 
                loss_G_L1 * self.config.loss.lambda_l1 + 
                loss_G_SOBEL * self.config.loss.lambda_sobel + 
                loss_G_LAP * self.config.loss.lambda_laplacian) 
        return { # Return generator losses
            'loss_G': loss_G,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G_SOBEL': loss_G_SOBEL,
            'loss_G_LAP': loss_G_LAP
        }
    
    def backward_G(self, loss_G: torch.Tensor) -> None:
        '''Backward step for the generator. Computes and applies gradients from
        generator loss and updates optimizer scaling accordingly.
        '''
        self.opt_gen.zero_grad() # Clear generator optimizer loss gradients
        self.g_scaler.scale(loss_G).backward() # Scale loss and compute gradient
        self.g_scaler.step(self.opt_gen) # Optimizer step
        self.g_scaler.update() # Update scaling factor

    def update_schedulers(self) -> None:
        '''Updates model schedulers'''
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
        losses_G = self.forward_G(x, y, y_fake)
        self.backward_G(losses_G['loss_G'])
        
        if idx % self.config.visualizer.update_frequency == 0:
            self.update_display(x, y, y_fake, idx, losses_G, losses_D)
        
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