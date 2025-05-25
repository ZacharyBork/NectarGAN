import time

from pix2pix_graphical.trainers.pix2pix_trainer import Pix2pixTrainer

if __name__ == "__main__":
    trainer = Pix2pixTrainer()
    trainer.build_output_directory() # Build experiment output directory
    trainer.export_config() # Export JSON config file to output directory

    cfg = trainer.config

    epoch_count = cfg.train.num_epochs + cfg.train.num_epochs_decay
    for epoch in range(epoch_count):
        begin_epoch = time.perf_counter() # Get epoch start time
        trainer.train(epoch) # Train generator and discriminator
        
        if epoch == epoch_count-1:
            trainer.save_checkpoint() # Always save model after final epoch
        elif cfg.save.save_model and epoch % cfg.save.model_save_rate == 0:
            trainer.save_checkpoint() # Save intermediate checkpoints
        
        if cfg.save.save_examples and epoch % cfg.save.example_save_rate == 0:
            trainer.save_examples() # Save example images if enabled

        end_epoch = time.perf_counter() # Get epoch end time
        trainer.print_end_of_epoch(begin_epoch, end_epoch)

