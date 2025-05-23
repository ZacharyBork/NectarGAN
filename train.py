import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import pathlib
import time

from utils import utility
from utils import scheduler
from models.generator_model import Generator
from models.discriminator_model import Discriminator
from models.loss import SobelLoss, LaplacianLoss

class Trainer():
    def __init__(self, config_filepath: str=None):
        # Get configuration data
        self.config = utility.get_config_data(config_filepath)

        self.init_networks()      # Init generator and discriminator
        self.build_dataloaders()  # Get datasets and create data loaders
        self.define_gradscalers() # Create gradient scalers
        self.init_losses()        # Init loss functions
        
        if self.config['LOAD']['CONTINUE_TRAIN']: # Load checkpoint if applicable
            utility.load_checkpoint(self.config, self.gen, self.opt_gen, self.disc, self.opt_disc)

    def init_networks(self):
        '''Initializes generator and discriminator, and also initializes
        an optimizer and a learning rate scheduler for each.
        '''
        # Init Generator
        self.gen = Generator( 
            input_size=self.config['COMMON']['CROP_SIZE'], 
            in_channels=self.config['COMMON']['INPUT_NC']).to(self.config['COMMON']['DEVICE'])
        # Init optimizer for generator
        self.opt_gen = optim.Adam( 
            self.gen.parameters(), 
            lr=self.config['TRAIN']['LEARNING_RATE'], 
            betas=(self.config['OPTIMIZER']['BETA1'], 0.999))
        # Init LR scheduler for generator
        self.gen_lr_scheduler = scheduler.LRScheduler( 
            self.opt_gen,
            self.config['TRAIN']['NUM_EPOCHS'],
            self.config['TRAIN']['NUM_EPOCHS_DECAY'])

        # Init discriminator
        self.disc = Discriminator(
            in_channels=self.config['COMMON']['INPUT_NC'],
            base_channels=self.config['DISCRIMINATOR']['BASE_CHANNELS_D'],
            n_layers=self.config['DISCRIMINATOR']['N_LAYERS_D'],
            max_channels=self.config['DISCRIMINATOR']['MAX_CHANNELS_D']).to(self.config['COMMON']['DEVICE'])
        # Init optimizer for discriminator
        self.opt_disc = optim.Adam(
            self.disc.parameters(), 
            lr=self.config['TRAIN']['LEARNING_RATE'], 
            betas=(self.config['OPTIMIZER']['BETA1'], 0.999))
        # Init LR scheduler for discriminator
        self.disc_lr_scheduler = scheduler.LRScheduler(
            self.opt_disc,
            self.config['TRAIN']['NUM_EPOCHS'],
            self.config['TRAIN']['NUM_EPOCHS_DECAY'])     
        
    def init_plotting(self):
        # Temporary numpy plotting for model validation
        plt.ion()
        image_data = np.random.rand(512, 512)
        self.fig, self.axes = plt.subplots(1, 3)

        names = ['real_A', 'fake_B', 'real_B']#, 'L1_response', 'SOBEL_response', 'LAP_response']
        self.ax_data = []
        for i, name in enumerate(names):
            # ax = self.axes[0 if i<3 else 1][i%3]
            ax = self.axes[i]
            ax.axis('off')
            ax.set_title(name)
            data = ax.imshow(image_data)
            self.ax_data.append(data)

    def build_dataloaders(self):
        '''Builds dataloaders for training and validation datasets.'''
        self.train_loader = utility.build_dataloader(self.config, 'train')
        self.val_loader = utility.build_dataloader(self.config, 'val')

    def define_gradscalers(self):
        '''Defines gradient scalers for generator and discriminator
        based on current selected device.
        '''
        if self.config['COMMON']['DEVICE'] == 'cpu':
            self.g_scaler = self.d_scaler = torch.amp.GradScaler('cpu')
        else: self.g_scaler = self.d_scaler = torch.amp.GradScaler('cuda')

    def init_losses(self):
        '''Initializes model loss functions.'''
        self.BCE = nn.BCEWithLogitsLoss().to(self.config['COMMON']['DEVICE'])
        self.L1_LOSS = nn.L1Loss().to(self.config['COMMON']['DEVICE'])
        self.SOBEL_LOSS = SobelLoss().to(self.config['COMMON']['DEVICE'])
        self.LAP_LOSS = LaplacianLoss().to(self.config['COMMON']['DEVICE'])

    def build_output_directory(self):
        '''Builds an output directory structure for the experiment.'''
        self.experiment_dir, self.examples_dir = utility.build_experiment_directory(
            self.config['COMMON']['OUTPUT_DIRECTORY'], 
            self.config['COMMON']['EXPERIMENT_NAME'], 
            self.config['LOAD']['CONTINUE_TRAIN'])
        
    def export_config(self):
        '''Exports a versioned config JSON file to the experiment output directory.'''
        utility.export_training_config(self.config, self.experiment_dir)

    def update_display(self, x, y, y_fake, current_epoch, idx, losses_G, losses_D):
        '''Function to update loss and image displays during training.'''
        utility.print_losses( # Print current loss values
            current_epoch, idx, 
            losses={
                'G_GAN': round(losses_G['loss_G_GAN'].item(), 2),
                'G_L1': round(losses_G['loss_G_L1'].item(), 2),
                'G_SOBEL': round(losses_G['loss_G_SOBEL'].item(), 2),
                'G_LAP': round(losses_G['loss_G_LAP'].item(), 2),
                'D_real': round(losses_D['loss_D_real'].item(), 2),
                'D_fake': round(losses_D['loss_D_fake'].item(), 2)
            }
        )

        # Convert x, y, y_fake tensors to images
        vis_x, vis_y = utility.tensor_to_image(x), utility.tensor_to_image(y)
        vis_y_fake = utility.tensor_to_image(y_fake)
        visualizers = [vis_x, vis_y_fake, vis_y]

        # Update numpy plotting
        for i, vis in enumerate(visualizers):
            try: self.ax_data[i].set_data(vis)
            except: continue
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def print_end_of_epoch(self, index, end_epoch, begin_epoch):
        print(f'(End of epoch {index}) Time: {end_epoch - begin_epoch:.2f} seconds', flush=True)
        gen_lr_step = self.gen_lr_scheduler.get_lr()
        print(f'Learning rate (G): {gen_lr_step[0]} => {gen_lr_step[1]}')
        disc_lr_step = self.disc_lr_scheduler.get_lr()
        print(f'Learning rate (D): {disc_lr_step[0]} => {disc_lr_step[1]}')

    def compute_loss_D(self, x, y, y_fake):
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

    def backward_D(self, loss_D):
        '''Backward step for the discriminator. Computes and applies gradients from
        discriminator loss and updates optimizer scaling accordingly.
        '''
        self.disc.zero_grad() # Clear loss gradients
        self.d_scaler.scale(loss_D).backward() # Scale loss and compute gradient
        self.d_scaler.step(self.opt_disc) # Optimizer step
        self.d_scaler.update() # Update scaling factor
        
    def compute_loss_G(self, lcon, x, y, y_fake):
        '''Forward step for generator. Run each iteration to compute
        generator loss.
        '''
        with torch.amp.autocast('cuda'):
            D_fake = self.disc(x, y_fake) # Get prediction from discriminator
            loss_G_GAN = self.BCE(D_fake, torch.ones_like(D_fake))
            loss_G_L1 = self.L1_LOSS(y_fake, y)
            loss_G_SOBEL = self.SOBEL_LOSS(y_fake, y) if lcon['DO_SOBEL_LOSS'] else torch.zeros_like(loss_G_L1)
            loss_G_LAP = self.LAP_LOSS(y_fake, y) if lcon['DO_LAPLACIAN_LOSS'] else torch.zeros_like(loss_G_L1)
            loss_G = ( # Calculate final generator loss
                loss_G_GAN + 
                loss_G_L1 * self.config['LOSS']['LAMBDA_L1'] + 
                loss_G_SOBEL * self.config['LOSS']['LAMBDA_SOBEL'] + 
                loss_G_LAP * self.config['LOSS']['LAMBDA_LAPLACIAN']) 
        return { # Return generator losses
            'loss_G': loss_G,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G_SOBEL': loss_G_SOBEL,
            'loss_G_LAP': loss_G_LAP
        }
    
    def backward_G(self, loss_G):
        '''Backward step for the generator. Computes and applies gradients from
        generator loss and updates optimizer scaling accordingly.
        '''
        self.opt_gen.zero_grad() # Clear generator optimizer loss gradients
        self.g_scaler.scale(loss_G).backward() # Scale loss and compute gradient
        self.g_scaler.step(self.opt_gen) # Optimizer step
        self.g_scaler.update() # Update scaling factor

    def update_schedulers(self):
        '''Updates model schedulers'''
        self.disc_lr_scheduler.step() # Discriminator LR scheduler
        self.gen_lr_scheduler.step()  # Generator LR scheduler

    def train(self, current_epoch:int):
        '''Training function for the generator and discriminator networks.
        
        Loops over each batch of (x, y) pairs in training set. Generator first makes
        a fake image (y_fake). Then losses are computed for discriminator based on x, 
        y, and y fake. Then losses are computed for generator based on discriminator
        output and additional loss functions. Losses get applied to both networks,
        then we print losses if applicable this iteration and move to next batch.
        '''
        lcon = self.config['LOSS'] # Get loss config settings
        for idx, (x, y) in enumerate(self.train_loader):
            # Get (x, y) of image[idx] from training dataset
            x, y = x.to(self.config['COMMON']['DEVICE']), y.to(self.config['COMMON']['DEVICE'])

            # Generate fake image
            with torch.amp.autocast('cuda'):
                y_fake = self.gen(x)

            # Get discriminator losses, apply gradients
            losses_D = self.compute_loss_D(x, y, y_fake)
            self.backward_D(losses_D['loss_D'])
            
            # Get generator losses, apply gradients
            losses_G = self.compute_loss_G(lcon, x, y, y_fake)
            self.backward_G(losses_G['loss_G'])
            
            if idx % self.config['MISC']['UPDATE_FREQUENCY'] == 0:
                self.update_display(x, y, y_fake, current_epoch, idx, losses_G, losses_D)

        self.update_schedulers() # Update schedulers after training

    def start(self):
        '''Runs a pix2pix training loop based on the config file provided during
        init, or default if no config path provided (/pix2pix-graphical/config.json)
        '''
        self.init_plotting() # Init numpy plotting
        plt.show() # Show pyplot

        self.build_output_directory() # Build experiment output directory
        self.export_config() # Export JSON config file to output directory

        epoch_count = self.config['TRAIN']['NUM_EPOCHS'] + self.config['TRAIN']['NUM_EPOCHS_DECAY']
        for epoch in range(epoch_count):
            begin_epoch = time.perf_counter()

            index = epoch + 1 + self.config['LOAD']['LOAD_EPOCH'] if self.config['LOAD']['CONTINUE_TRAIN'] else epoch + 1
            self.train(index)
            
            if epoch == epoch_count-1:
                # Always save model after final epoch
                checkpoint_name = 'final_net{}.pth.tar'
                utility.save_checkpoint(
                    self.gen, self.opt_gen, output_path=pathlib.Path(self.experiment_dir, checkpoint_name.format(str(index), 'G')))
                utility.save_checkpoint(
                    self.disc, self.opt_disc, output_path=pathlib.Path(self.experiment_dir, checkpoint_name.format(str(index), 'D')))
            elif self.config['SAVE']['SAVE_MODEL'] and epoch % self.config['SAVE']['MODEL_SAVE_RATE'] == 0:
                # Save intermediate checkpoints if applicable
                checkpoint_name = 'epoch{}_net{}.pth.tar'
                utility.save_checkpoint(
                    self.gen, self.opt_gen, output_path=pathlib.Path(self.experiment_dir, checkpoint_name.format(str(index), 'G')))
                utility.save_checkpoint(
                    self.disc, self.opt_disc, output_path=pathlib.Path(self.experiment_dir, checkpoint_name.format(str(index), 'D')))
            
            if self.config['SAVE']['SAVE_EXAMPLES'] and epoch % self.config['SAVE']['EXAMPLE_SAVE_RATE'] == 0:
                utility.save_examples(self.config, self.gen, self.val_loader, index, output_directory=self.examples_dir)

            end_epoch = time.perf_counter()
            self.print_end_of_epoch(index, end_epoch, begin_epoch)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.start()

