import pathlib
from typing import Any
from os import PathLike

from PySide6.QtCore import QObject, Signal, Slot

from nectargan.testers.tester import Tester

class TesterWorker(QObject, Tester):
    finished = Signal(pathlib.Path)
    progress = Signal(int)

    def __init__(
            self, 
            config: PathLike | dict[str, Any] | None,
            image_count: int=30,
            experiment_dir: PathLike | None=None,
            dataroot: PathLike | None=None,
            load_epoch: int | None=None
        ) -> None:
        super().__init__(
            config=config, 
            experiment_dir=experiment_dir,
            dataroot=dataroot,
            load_epoch=load_epoch)
        self.image_count = image_count

    @Slot()
    def run(self) -> None:
        self.build_test_output_directory()
        self.export_base_test_log()
        self.init_loss_functions()
        
        indices = self.build_lookup_indices(self.image_count)
        for i, idx in enumerate(indices):
            self._test_step(i=i, idx=idx)
            self.progress.emit(i+1)
        self.finished.emit(self.output_dir)

