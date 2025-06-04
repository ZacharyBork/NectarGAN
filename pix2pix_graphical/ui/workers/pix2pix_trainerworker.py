import time
from typing import Any
from os import PathLike

import torch
from PySide6.QtCore import QObject, Signal, Slot

from pix2pix_graphical.trainers.pix2pix_trainer import Pix2pixTrainer

class TrainerWorker(QObject, Pix2pixTrainer):
    finished = Signal()
    epoch_progress = Signal(int)
    train_progress = Signal(list)
    epoch_time = Signal(float)
    tensors = Signal(torch.Tensor)
    log = Signal(str)
    cancelled = Signal()
    waiting = Signal(int)

    def __init__(
            self, 
            config: PathLike | dict[str, Any] | None,
            loss_subspec: str='extended+vgg',
            log_losses: bool=False
        ) -> None:
        super().__init__(
            config=config, 
            loss_subspec=loss_subspec,
            log_losses=log_losses)
        self._is_paused: bool = False
        self._is_interrupted: bool = False
        self._update_frequency: int = 50

    def pause(self):
        self.log.emit('Training Paused...')
        self._is_paused = True

    def unpause(self):
        self.log.emit('Resuming Training...')
        self._is_paused = False

    def stop(self):
        self._is_interrupted = True
    
    def change_update_frequency(self, value: int):
        self._update_frequency = value

    def pause_training(
            self, 
            polling_rate: float=0.5,
            counter_max: int=999,
        ) -> None:
        counter = 0
        while self._is_paused:
            if self._is_interrupted:
                self.cancelled.emit()
                return
            self.waiting.emit(counter)
            counter = (counter + 1) % counter_max
            time.sleep(polling_rate)

    def train_step(
            self, 
            x: torch.Tensor, 
            y: torch.Tensor, 
            idx: int,
            **kwargs: Any
        ) -> torch.Tensor:
        with torch.amp.autocast('cuda'): 
            y_fake = self.gen(x)

        # Get discriminator losses, apply gradients
        losses_D = self.forward_D(x, y, y_fake)
        self.backward_D(losses_D['loss_D'])
        
        # Get generator losses, apply gradients
        loss_G = self.forward_G(x, y, y_fake)
        self.backward_G(loss_G)

        return y_fake

    @Slot()
    def run(self):
        self.config.dataloader.num_workers = 0
        self.config.visualizer.visdom.enable = False
        self.cfg = self.config

        epoch_counts = self.cfg.train.generator.learning_rate
        self.epoch_count = epoch_counts.epochs + epoch_counts.epochs_decay

        for epoch in range(self.epoch_count):    
            start_time = time.perf_counter()

            if self.config.train.load.continue_train:
                self.current_epoch = 1 + epoch + self.config.train.load.load_epoch
            else: self.current_epoch = epoch + 1
            
            self.on_epoch_start() 
            for idx, (x, y) in enumerate(self.train_loader):
                if self._is_interrupted: 
                    self.log.emit('Training Canceled. Stopping...')
                    self.cancelled.emit()
                    return
                if self._is_paused: self.pause_training()


                x, y = x.to(self.device), y.to(self.device)
                y_fake = self.train_step(x, y, idx)

                if idx % self._update_frequency == 0:
                    self.tensors.emit((
                        x.detach().cpu(), 
                        y.detach().cpu(), 
                        y_fake.detach().cpu()))
                    self.log.emit(self.loss_manager.print_losses(epoch, idx, capture=True))

                self.epoch_progress.emit(int((idx / len(self.train_loader)) * 100.0))

            self.on_epoch_end() 
            
            end_time = time.perf_counter()
            self.last_epoch_time = end_time-start_time
            self.train_progress.emit([epoch, self.last_epoch_time])
            # self.epoch_time.emit(self.last_epoch_time)
            self.log.emit(self.print_end_of_epoch(capture=True))
            

            if epoch == self.epoch_count-1:
                self.log.emit(self.save_checkpoint(capture=True))
            elif self.cfg.save.save_model and (epoch+1) % self.cfg.save.model_save_rate == 0:
                self.log.emit(self.save_checkpoint(capture=True))

            if (self.cfg.save.save_examples
                and (epoch+1) % self.cfg.save.example_save_rate == 0):
                self.log.emit(self.save_examples(capture=True))
        self.finished.emit()

            
