from pathlib import Path
from typing import Any
from concurrent.futures import ProcessPoolExecutor, wait
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

import pix2pix_graphical.toolbox.utils.common as utils

class SignalHandler(QObject):
    progress = Signal(int)
    finished = Signal()

class UtilityWorker(QObject):
    def __init__(self, args: Any, signalhandler: SignalHandler) -> None:
        super().__init__()
        self.args = args
        self.signalhandler = signalhandler
        self.progress_total = len(self.args)
        self.progress_finished = 0

    def send_progress_signal(self, future) -> None:
        self.progress_finished += 1
        self.signalhandler.progress.emit(
            int((self.progress_finished / self.progress_total) * 100.0))
        if self.progress_finished == self.progress_total:
            self.signalhandler.finished.emit()    

    @Slot()
    def pair_images(self) -> None:
        executor = ProcessPoolExecutor()
        for arg in self.args:
            future = executor.submit(utils.concat_images, arg)
            future.add_done_callback(self.send_progress_signal)

    @Slot()
    def sort_images(self) -> None:
        executor = ProcessPoolExecutor()
        futures = []
        for arg in self.args:
            future = executor.submit(utils.get_image_value, arg)
            future.add_done_callback(self.send_progress_signal)
            futures.append(future)
        wait(futures)
        values = [i.result() for i in futures]
        values = sorted(values, key=lambda x : x[0])
        for i, value in enumerate(values):
            path = value[1]
            new_path = Path(path.parent.resolve(), f'{i}_{path.name}')
            path.rename(new_path)
        self.signalhandler.finished.emit()  
