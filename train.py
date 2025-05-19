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

def train(config: dict, current_epoch:int, disc: Discriminator, gen: Generator, loader: torch.utils.data.DataLoader, 
          opt_disc: optim.Adam, opt_gen: optim.Adam, l1: nn.L1Loss, bce: nn.BCEWithLogitsLoss, 
          sobel: SobelLoss, g_scaler: torch.amp.GradScaler, d_scaler: torch.amp.GradScaler):
    '''Training function for generator and discriminator.'''
    
    for idx, (x, y) in enumerate(loader):
        x, y = x.to(config['DEVICE']), y.to(config['DEVICE'])

        # Train Discriminator
        with torch.amp.autocast('cuda'):
            y_fake = gen(x)
            D_real = disc(x, y)
            D_fake = disc(x, y_fake.detach())
            loss_D_real = bce(D_real, torch.ones_like(D_real))
            loss_D_fake = bce(D_fake, torch.zeros_like(D_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        
        disc.zero_grad()
        d_scaler.scale(loss_D).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator
        with torch.amp.autocast('cuda'):
            D_fake = disc(x, y_fake)
            loss_G_GAN = bce(D_fake, torch.ones_like(D_fake))
            loss_G_L1 = l1(y_fake, y) * config['LAMBDA_L1']
            loss_G_SOBEL = sobel(y_fake, y) * config['LAMBDA_SOBEL']
            loss_G = loss_G_GAN + loss_G_L1 + loss_G_SOBEL

        opt_gen.zero_grad()
        g_scaler.scale(loss_G).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % config['UPDATE_FREQUENCY'] == 0:
            losses = {
                'G_GAN': round(loss_G_GAN.item(), 2),
                'G_L1': round(loss_G_L1.item(), 2),
                'G_SOBEL': round(loss_G_SOBEL.item(), 2),
                'D_real': round(loss_D_real.item(), 2),
                'D_fake': round(loss_D_fake.item(), 2)
            }
            utility.print_losses(current_epoch, idx, losses)
            

def main():
    # Get config data
    config = utility.get_config_data()

    # Build experiment output directory and export config JSON
    experiment_dir, examples_dir = utility.build_experiment_directory(
        config['OUTPUT_DIRECTORY'], config['EXPERIMENT_NAME'], config['CONTINUE_TRAIN'])
    utility.export_training_config(config, experiment_dir)

    # Init generator and gen optimizer
    gen = Generator(in_channels=config['INPUT_NC']).to(config['DEVICE'])
    opt_gen = optim.Adam(gen.parameters(), lr=config['LEARNING_RATE'], betas=(config['BETA1'], 0.999))

    # Init discriminator and disc optimizer
    disc = Discriminator(in_channels=config['INPUT_NC']).to(config['DEVICE'])
    opt_disc = optim.Adam(disc.parameters(), lr=config['LEARNING_RATE'], betas=(config['BETA1'], 0.999))

    # Load checkpoint if applicable
    if config['CONTINUE_TRAIN']:
        utility.load_checkpoint(config, gen, opt_gen, disc, opt_disc)

    # Get datasets and create data loaders
    train_loader = utility.build_dataloader(config, 'train')
    val_loader = utility.build_dataloader(config, 'val')

    # Create gradient scalers
    g_scaler = torch.amp.GradScaler('cuda')
    d_scaler = torch.amp.GradScaler('cuda')

    # Init losses
    BCE = nn.BCEWithLogitsLoss().to(config['DEVICE'])
    L1_LOSS = nn.L1Loss().to(config['DEVICE'])
    SOBEL_LOSS = SobelLoss().to(config['DEVICE'])

    # Training loop
    for epoch in range(config['NUM_EPOCHS']):
        index = epoch + 1 + config['LOAD_EPOCH'] if config['CONTINUE_TRAIN'] else epoch + 1
        train(config, index, disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, SOBEL_LOSS, g_scaler, d_scaler)
        
        if config['SAVE_MODEL'] and epoch % config['MODEL_SAVE_RATE'] == 0:
            checkpoint_name = 'epoch{}_net{}.pth.tar'
            utility.save_checkpoint(
                gen, opt_gen, output_path=pathlib.Path(experiment_dir, checkpoint_name.format(str(index), 'G')))
            utility.save_checkpoint(
                disc, opt_disc, output_path=pathlib.Path(experiment_dir, checkpoint_name.format(str(index), 'D')))

        utility.save_examples(config, gen, val_loader, index, output_directory=examples_dir)

if __name__ == "__main__":
    main()

