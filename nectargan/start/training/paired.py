from nectargan.start.torch_check import validate_torch
validate_torch()

import argparse
from nectargan.trainers.pix2pix_trainer import Pix2pixTrainer

def init_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--config_file', type=str, default=None, 
        help='Path to training config file.')
    parser.add_argument(
        '-lss', '--loss_subspec', type=str, default='basic', 
        help='Loss subspec for pix2pix LossManager. ("basic", "extended")')
    parser.add_argument(
        '-log', '--log_losses', action='store_true',
        help='Enables loss history logging.')
    return parser.parse_args()

def main():
    args = init_cli()

    trainer = Pix2pixTrainer(
        config=args.config_file, 
        loss_subspec=args.loss_subspec,
        log_losses=args.log_losses)
    cfg = trainer.config

    # Build epoch count from generator LR schedule
    epoch_counts = cfg.train.generator.learning_rate
    epoch_count = epoch_counts.epochs + epoch_counts.epochs_decay
    
    for epoch in range(epoch_count):
        trainer.train_paired( # Train generator and discriminator
            epoch, 
            callback_kwargs={ 'on_epoch_start': {'print_train_start': True} }) 
        
        if epoch == epoch_count-1:
            trainer.save_checkpoint() # Always save model after final epoch
        elif cfg.save.save_model and (epoch+1) % cfg.save.model_save_rate == 0:
            trainer.save_checkpoint() # Save intermediate checkpoints

        if (cfg.save.save_examples
            and (epoch+1) % cfg.save.example_save_rate == 0):
            trainer.save_examples() # Save example images if applicable

        trainer.print_end_of_epoch()

if __name__ == "__main__":
    main()

