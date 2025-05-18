import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tdqm import tdqm

from utility import utils
from utility.dataset import Pix2pixDataset
from . import config

from models.generator_model import Generator
from models.discriminator_model import Discriminator

def train_fn(disc, gen, loader, opt_disc, opt_gen, l1, bce, g_scaler, d_scaler):
    loop = tdqm(loader, leave=True)
    
    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_fake = disc(x, y_fake.detach())
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) * 0.5
        
        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1(y_fake, y) * config.LAMBDA_L1
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

def main():
    # Init generator and gen optimizer
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    # Init discriminator and disc optimizer
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    # Init losses
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    # Load checkpoint if applicable
    if config.LOAD_MODEL:
        utils.load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        utils.load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

    # Get training dataset and create data loader
    train_dataset = Pix2pixDataset(root_dir='/path/to/training/dataset')
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    # Get validation dataset and create data loader
    val_dataset = Pix2pixDataset(root_dir='/path/to/validation/dataset')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Create gradient scalers
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler)

        if config.SAVE_MODEL and epoch % config.MODEL_SAVE_RATE == 0:
            utils.save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            utils.save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        utils.save_examples(gen, val_loader, epoch, output_directory='evaluation')



if __name__ == "__main__":
    main()