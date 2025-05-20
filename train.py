import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pathlib

from utils import utility
from utils.dataset import Pix2pixDataset

from models.generator_model import Generator
from models.discriminator_model import Discriminator
from models.networks import SobelLoss

class Trainer():
    def __init__(self, config_filepath: str=None):
        self.config = utility.get_config_data(config_filepath)

        # Init generator and gen optimizer
        self.gen = Generator(in_channels=self.config['INPUT_NC']).to(self.config['DEVICE'])
        self.opt_gen = optim.Adam(self.gen.parameters(), lr=self.config['LEARNING_RATE'], betas=(self.config['BETA1'], 0.999))

        # Init discriminator and disc optimizer
        self.disc = Discriminator(
            in_channels=self.config['INPUT_NC'],
            base_channels=self.config['BASE_CHANNELS_D'],
            n_layers=self.config['N_LAYERS_D'],
            max_channels=self.config['MAX_CHANNELS_D']).to(self.config['DEVICE'])
        self.opt_disc = optim.Adam(self.disc.parameters(), lr=self.config['LEARNING_RATE'], betas=(self.config['BETA1'], 0.999))

        # Get datasets and create data loaders
        self.train_loader = utility.build_dataloader(self.config, 'train')
        self.val_loader = utility.build_dataloader(self.config, 'val')

        # Create gradient scalers
        self.g_scaler = torch.amp.GradScaler('cuda')
        self.d_scaler = torch.amp.GradScaler('cuda')

        # Init losses
        self.BCE = nn.BCEWithLogitsLoss().to(self.config['DEVICE'])
        self.L1_LOSS = nn.L1Loss().to(self.config['DEVICE'])
        self.SOBEL_LOSS = SobelLoss().to(self.config['DEVICE'])

        # Load checkpoint if applicable
        if self.config['CONTINUE_TRAIN']:
            utility.load_checkpoint(self.config, self.gen, self.opt_gen, self.disc, self.opt_disc)

    def init_plotting(self):
        # Temporary numpy plotting for model validation

        plt.ion()
        image_data = np.random.rand(512, 512)
        self.fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        ax1.axis('off')
        ax1.set_title('real_A')
        self.ax1_data = ax1.imshow(image_data)
        ax2.axis('off')
        ax2.set_title('fake_B')
        self.ax2_data = ax2.imshow(image_data)
        ax3.axis('off')
        ax3.set_title('real_B')
        self.ax3_data = ax3.imshow(image_data)

    def build_output_directory(self):
        '''Builds an output directory structure for the experiment.'''
        self.experiment_dir, self.examples_dir = utility.build_experiment_directory(
            self.config['OUTPUT_DIRECTORY'], self.config['EXPERIMENT_NAME'], self.config['CONTINUE_TRAIN'])
        
    def export_config(self):
        '''Exports a versioned config JSON file to the experiment output directory.'''
        utility.export_training_config(self.config, self.experiment_dir)

    def train_D(self, x, y):
        '''Train discriminator and get loss values.'''
        with torch.amp.autocast('cuda'):
            y_fake = self.gen(x)
            D_real = self.disc(x, y)
            D_fake = self.disc(x, y_fake.detach())
            loss_D_real = self.BCE(D_real, torch.ones_like(D_real))
            loss_D_fake = self.BCE(D_fake, torch.zeros_like(D_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            
        self.disc.zero_grad()
        self.d_scaler.scale(loss_D).backward()
        self.d_scaler.step(self.opt_disc)
        self.d_scaler.update()
        return y_fake, loss_D_real, loss_D_fake
    
    def train_G(self, x, y, y_fake):
        '''Train generator and get loss values.'''
        with torch.amp.autocast('cuda'):
            D_fake = self.disc(x, y_fake)
            loss_G_GAN = self.BCE(D_fake, torch.ones_like(D_fake))
            loss_G_L1 = self.L1_LOSS(y_fake, y) * self.config['LAMBDA_L1']
            loss_G_SOBEL = self.SOBEL_LOSS(y_fake, y) * self.config['LAMBDA_SOBEL']
            loss_G = loss_G_GAN + loss_G_L1 + loss_G_SOBEL

        self.opt_gen.zero_grad()
        self.g_scaler.scale(loss_G).backward()
        self.g_scaler.step(self.opt_gen)
        self.g_scaler.update()
        return loss_G_GAN, loss_G_L1, loss_G_SOBEL

    def train_epoch(self, current_epoch:int):
        '''Training function for generator and discriminator.'''
        plt.show()
        for idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.config['DEVICE']), y.to(self.config['DEVICE'])

            # Train discriminator, get losses
            y_fake, loss_D_real, loss_D_fake = self.train_D(x, y)
            
            # Train generator, get losses
            loss_G_GAN, loss_G_L1, loss_G_SOBEL = self.train_G(x, y, y_fake)

            if idx % self.config['UPDATE_FREQUENCY'] == 0:
                vis_x = utility.tensor_to_image(x)
                vis_y = utility.tensor_to_image(y)
                vis_y_fake = utility.tensor_to_image(y_fake)
                losses = {
                    'G_GAN': round(loss_G_GAN.item(), 2),
                    'G_L1': round(loss_G_L1.item(), 2),
                    'G_SOBEL': round(loss_G_SOBEL.item(), 2),
                    'D_real': round(loss_D_real.item(), 2),
                    'D_fake': round(loss_D_fake.item(), 2)
                }
                utility.print_losses(current_epoch, idx, losses)

                # Update numpy plotting
                self.ax1_data.set_data(vis_x)
                self.ax2_data.set_data(vis_y_fake)
                self.ax3_data.set_data(vis_y)
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

    def start(self):
        '''Start pix2pix training loop.'''
        self.init_plotting() # Init numpy plotting

        self.build_output_directory() # Build experiment output directory
        self.export_config() # Export JSON config file to output directory
        for epoch in range(self.config['NUM_EPOCHS']):
            index = epoch + 1 + self.config['LOAD_EPOCH'] if self.config['CONTINUE_TRAIN'] else epoch + 1
            self.train_epoch(index)
            
            if self.config['SAVE_MODEL'] and epoch % self.config['MODEL_SAVE_RATE'] == 0:
                checkpoint_name = 'epoch{}_net{}.pth.tar'
                utility.save_checkpoint(
                    self.gen, self.opt_gen, output_path=pathlib.Path(self.experiment_dir, checkpoint_name.format(str(index), 'G')))
                utility.save_checkpoint(
                    self.disc, self.opt_disc, output_path=pathlib.Path(self.experiment_dir, checkpoint_name.format(str(index), 'D')))

            utility.save_examples(self.config, self.gen, self.val_loader, index, output_directory=self.examples_dir)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.start()

