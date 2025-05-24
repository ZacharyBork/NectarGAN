import time
import pathlib

from pix2pix_graphical.utils import utility
from pix2pix_graphical.trainers.trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer()
    trainer.build_output_directory() # Build experiment output directory
    trainer.export_config() # Export JSON config file to output directory

    epoch_count = trainer.config.train.num_epochs + trainer.config.train.num_epochs_decay
    for epoch in range(epoch_count):
        begin_epoch = time.perf_counter()

        index = epoch + 1 + trainer.config.load.load_epoch if trainer.config.load.continue_train else epoch + 1
        trainer.train(index)
        
        if epoch == epoch_count-1:
            # Always save model after final epoch
            checkpoint_name = 'final_net{}.pth.tar'
            utility.save_checkpoint(
                trainer.gen, trainer.opt_gen, output_path=pathlib.Path(trainer.experiment_dir, checkpoint_name.format(str(index), 'G')))
            utility.save_checkpoint(
                trainer.disc, trainer.opt_disc, output_path=pathlib.Path(trainer.experiment_dir, checkpoint_name.format(str(index), 'D')))
        elif trainer.config.save.save_model and epoch % trainer.config.save.model_save_rate == 0:
            # Save intermediate checkpoints if applicable
            checkpoint_name = 'epoch{}_net{}.pth.tar'
            utility.save_checkpoint(
                trainer.gen, trainer.opt_gen, output_path=pathlib.Path(trainer.experiment_dir, checkpoint_name.format(str(index), 'G')))
            utility.save_checkpoint(
                trainer.disc, trainer.opt_disc, output_path=pathlib.Path(trainer.experiment_dir, checkpoint_name.format(str(index), 'D')))
        
        if trainer.config.save.save_examples and epoch % trainer.config.save.example_save_rate == 0:
            utility.save_examples(trainer.config, trainer.gen, trainer.val_loader, index, output_directory=trainer.examples_dir)

        end_epoch = time.perf_counter()
        trainer.print_end_of_epoch(index, end_epoch, begin_epoch)

