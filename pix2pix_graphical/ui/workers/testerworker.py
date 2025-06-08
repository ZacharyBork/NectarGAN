from typing import Any
from os import PathLike

from PySide6.QtCore import QObject, Signal, Slot

from pix2pix_graphical.testers.tester import Tester

class TesterWorker(QObject, Tester):
    finished = Signal()

    def __init__(
            self, 
            config: PathLike | dict[str, Any] | None=None
        ) -> None:
        super().__init__(config=config)

    @Slot()
    def run(self):
        self.run_test(image_count=30, silent=True)

        self.finished.emit()

