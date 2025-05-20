from PySide6.QtCore import QThread, Signal
import numpy as np
import importlib.util
import sys
import pathlib

from .train import Trainer

class TrainerWorker(QThread):
    loss_updated = Signal(float)
    image_updated = Signal(np.ndarray, np.ndarray, np.ndarray)

    def __init__(self, config):
        super().__init__()
        self.trainer = Trainer(config)
        self.stop_flag = False

    def run(self):
        for epoch in range(self.trainer.config['epochs']):
            if self.stop_flag:
                break
            for batch in self.trainer.dataloader:
                loss = self.trainer.train_step(batch)
                self.loss_updated.emit(float(loss))

                if self.should_emit_image():
                    input_img, output_img, target_img = self.trainer.get_visualization()
                    self.image_updated.emit(input_img, output_img, target_img)

    def stop(self):
        self.stop_flag = True