from PySide6.QtCore import QObject, Signal, Slot
import time

from pix2pix_graphical.config.config_data import Config
from pix2pix_graphical.trainers.pix2pix_trainer import Pix2pixTrainer

class TrainerWorker(QObject, Pix2pixTrainer):
    finished = Signal()
    progress = Signal(int)
    cancelled = Signal()

    def __init__(self) -> None:
        super().__init__(
            config=None, 
            loss_subspec='extended+vgg',
            log_losses=False)
        self._is_interrupted = False

    def stop(self):
        self._is_interrupted = True

    @Slot()
    def run(self):
        self.config.dataloader.num_workers = 0
        self.cfg = self.config

        epoch_counts = self.cfg.train.generator.learning_rate
        self.epoch_count = epoch_counts.epochs + epoch_counts.epochs_decay

        for epoch in range(self.epoch_count):
            if self._is_interrupted:
                self.cancelled.emit()
                return
            
            start_time = time.perf_counter()

            if self.config.train.load.continue_train:
                self.current_epoch = 1 + epoch + self.config.train.load.load_epoch
            else: self.current_epoch = epoch + 1
            
            self.on_epoch_start() 
            for idx, (x, y) in enumerate(self.train_loader):
                if self._is_interrupted: 
                    print('Training stopped')
                    self.cancelled.emit()
                    return
                x, y = x.to(self.device), y.to(self.device)
                self.train_step(x, y, idx)
            self.on_epoch_end() 
            
            end_time = time.perf_counter()
            self.last_epoch_time = end_time-start_time

        if epoch == self.epoch_count-1:
            self.save_checkpoint() # Always save model after final epoch
        elif self.cfg.save.save_model and (epoch+1) % self.cfg.save.model_save_rate == 0:
            self.save_checkpoint() # Save intermediate checkpoints

        if (self.cfg.save.save_examples
            and (epoch+1) % self.cfg.save.example_save_rate == 0):
            self.save_examples() # Save example images if applicable

        self.print_end_of_epoch()
        self.progress.emit(int((epoch + 1) / self.epoch_count * 100))




        # self.trainer = Pix2pixTrainer(
        #     config=None, 
        #     loss_subspec='extended+vgg',
        #     log_losses=False)
        # self.config.dataloader.num_workers = 0
        # self.cfg = self.config

        # # Build epoch count from generator LR schedule
        # epoch_counts = self.cfg.train.generator.learning_rate
        # self.epoch_count = epoch_counts.epochs + epoch_counts.epochs_decay

        # for epoch in range(self.epoch_count):
        #     if self._is_interrupted:
        #         self.cancelled.emit()
        #         return
            
        #     result = self.trainer._train_paired_QThread(
        #         epoch, 
        #         is_interrupted=lambda: self._is_interrupted,
        #         callback_kwargs={})
        #     if result == -1:
        #         self.cancelled.emit()
        #         return
            
        #     if epoch == self.epoch_count-1:
        #         self.trainer.save_checkpoint() # Always save model after final epoch
        #     elif self.cfg.save.save_model and (epoch+1) % self.cfg.save.model_save_rate == 0:
        #         self.trainer.save_checkpoint() # Save intermediate checkpoints

        #     if (self.cfg.save.save_examples
        #         and (epoch+1) % self.cfg.save.example_save_rate == 0):
        #         self.trainer.save_examples() # Save example images if applicable

        #     self.trainer.print_end_of_epoch()
        #     self.progress.emit(int((epoch + 1) / self.epoch_count * 100))
