import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import pathlib

import json

from utils import utility
from models.generator_model import Generator
from models.discriminator_model import Discriminator
# from models.loss import SobelLoss, LaplacianLoss

# from models.generator_flat import Generator
# from models.discriminator_flat import Discriminator

class Trainer():
    def __init__(self, config_filepath: str=None):
        self.config = utility.get_config_data(config_filepath)
        print(f'Train Config: {pathlib.Path(config_filepath).as_posix()}')
        print(json.dumps(self.config, indent=2))

        # Init generator and gen optimizer
        self.gen = Generator(input_size=self.config['CROP_SIZE'], in_channels=self.config['INPUT_NC']).to(self.config['DEVICE'])
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
        # if self.config['DO_SOBEL_LOSS']:
        #     self.SOBEL_LOSS = SobelLoss().to(self.config['DEVICE'])
        # if self.config['DO_LAPLACIAN_LOSS']:
        #     self.LAP_LOSS = LaplacianLoss().to(self.config['DEVICE'])

        # Load checkpoint if applicable
        if self.config['CONTINUE_TRAIN']:
            utility.load_checkpoint(self.config, self.gen, self.opt_gen, self.disc, self.opt_disc)

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
            y_fake = self.gen(x) # Generate fake image
            D_real = self.disc(x, y) # Make prediction on real image
            D_fake = self.disc(x, y_fake.detach()) # Then on fake image
            loss_D_real = self.BCE(D_real, torch.ones_like(D_real)) # Get real image loss
            loss_D_fake = self.BCE(D_fake, torch.zeros_like(D_fake)) # Get fake image loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5 # Normalized loss gradient
            
        self.disc.zero_grad() # Clear loss gradients
        self.d_scaler.scale(loss_D).backward() # Scale loss and compute gradient
        self.d_scaler.step(self.opt_disc) # Optimizer step
        self.d_scaler.update() # Update scaling factor
        return y_fake, loss_D_real, loss_D_fake # Return losses
    
    def train_G(self, x, y, y_fake):
        '''Train generator and get loss values.'''
        with torch.amp.autocast('cuda'):
            D_fake = self.disc(x, y_fake) # Get prediction from discriminator
            loss_G_GAN = self.BCE(D_fake, torch.ones_like(D_fake)) # Get GAN loss
            loss_G_L1 = self.L1_LOSS(y_fake, y) * self.config['LAMBDA_L1'] # L1 loss
            
            # self.grad_l1 = torch.abs(y_fake - y)
            
            # if self.config['DO_SOBEL_LOSS']:
            #     loss_G_SOBEL = self.SOBEL_LOSS(y_fake, y) * self.config['LAMBDA_SOBEL'] # SOBEL loss
            # else: loss_G_SOBEL = torch.zeros_like(loss_G_L1)
            # if self.config['DO_LAPLACIAN_LOSS']:
            #     loss_G_LAP = self.LAP_LOSS(y_fake, y) * self.config['LAMBDA_LAPLACIAN'] # LAPLACIAN loss
            # else: loss_G_LAP = torch.zeros_like(loss_G_L1)
            loss_G = loss_G_GAN + loss_G_L1# + loss_G_SOBEL + loss_G_LAP # Calculate loss gradient

        self.opt_gen.zero_grad() # Clear generator optimizer loss gradients
        self.g_scaler.scale(loss_G).backward() # Scale loss and compute gradient
        self.g_scaler.step(self.opt_gen) # Optimizer step
        self.g_scaler.update() # Update scaling factor
        return loss_G_GAN, loss_G_L1, 0.0, 0.0 #, loss_G_SOBEL, loss_G_LAP # Return losses

    def train(self, current_epoch:int):
        '''Training function for generator and discriminator.'''
        for idx, (x, y) in enumerate(self.train_loader):
            # Get (x, y) of image[idx] from training dataset
            x, y = x.to(self.config['DEVICE']), y.to(self.config['DEVICE'])

            # Train discriminator, get losses
            y_fake, loss_D_real, loss_D_fake = self.train_D(x, y)
            
            # Train generator, get losses
            loss_G_GAN, loss_G_L1, loss_G_SOBEL, loss_G_LAP = self.train_G(x, y, y_fake)

            
            if idx % self.config['UPDATE_FREQUENCY'] == 0:
                # Ensure fake output is between -1 and 1
                # y_fake = torch.clamp(y_fake, -1.0, 1.0)

                vis_x = utility.tensor_to_image(x)
                vis_y = utility.tensor_to_image(y)
                vis_y_fake = utility.tensor_to_image(y_fake)
                #vis_l1_grad = self.grad_l1.detach().mean(dim=1)[0].cpu()
                losses = {
                    'G_GAN': round(loss_G_GAN.item(), 2),
                    'G_L1': round(loss_G_L1.item(), 2),
                    'G_SOBEL': 0.0, #round(loss_G_SOBEL.item(), 2),
                    'G_LAP': 0.0, #round(loss_G_LAP.item(), 2),
                    'D_real': round(loss_D_real.item(), 2),
                    'D_fake': round(loss_D_fake.item(), 2)
                }
                utility.print_losses(current_epoch, idx, losses)

                visualizers = [vis_x, vis_y_fake, vis_y]#, vis_l1_grad]

                # Update numpy plotting
                for i, vis in enumerate(visualizers):
                    try: self.ax_data[i].set_data(vis)
                    except: continue
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

    def start(self):
        '''Start pix2pix training loop.'''
        self.init_plotting() # Init numpy plotting
        plt.show() # Show pyplot

        self.build_output_directory() # Build experiment output directory
        self.export_config() # Export JSON config file to output directory
        for epoch in range(self.config['NUM_EPOCHS']):
            index = epoch + 1 + self.config['LOAD_EPOCH'] if self.config['CONTINUE_TRAIN'] else epoch + 1
            self.train(index)
            
            if epoch == self.config['NUM_EPOCHS']-1:
                # Always save model after final epoch
                checkpoint_name = 'final_net{}.pth.tar'
                utility.save_checkpoint(
                    self.gen, self.opt_gen, output_path=pathlib.Path(self.experiment_dir, checkpoint_name.format(str(index), 'G')))
                utility.save_checkpoint(
                    self.disc, self.opt_disc, output_path=pathlib.Path(self.experiment_dir, checkpoint_name.format(str(index), 'D')))
            elif self.config['SAVE_MODEL'] and epoch % self.config['MODEL_SAVE_RATE'] == 0:
                # Save intermediate checkpoints if applicable
                checkpoint_name = 'epoch{}_net{}.pth.tar'
                utility.save_checkpoint(
                    self.gen, self.opt_gen, output_path=pathlib.Path(self.experiment_dir, checkpoint_name.format(str(index), 'G')))
                utility.save_checkpoint(
                    self.disc, self.opt_disc, output_path=pathlib.Path(self.experiment_dir, checkpoint_name.format(str(index), 'D')))
            
            if self.config['SAVE_EXAMPLES'] and epoch % self.config['EXAMPLE_SAVE_RATE'] == 0:
                utility.save_examples(self.config, self.gen, self.val_loader, index, output_directory=self.examples_dir)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.start()

