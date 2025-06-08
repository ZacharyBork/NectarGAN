
from PySide6.QtCore import QObject, QThread
from PySide6.QtWidgets import QWidget

from pix2pix_graphical.ui.workers.testerworker import TesterWorker

class TesterHelper(QObject):
    def __init__(self, mainwidget: QWidget) -> None:
        super().__init__()
        self.mainwidget = mainwidget
        self.find = self.mainwidget.findChild

        self.test_thread: QThread | None=None
        self.worker: TesterWorker | None=None 

    def start_test(self) -> None:     
        self.test_thread = QThread() 
        self.worker = TesterWorker()
        self.worker.moveToThread(self.test_thread)

        self.test_thread.started.connect(self.worker.run)
        
        self.worker.finished.connect(self.test_thread.quit)
        self.worker.cancelled.connect(self.test_thread.quit)
        
        self.test_thread.finished.connect(self.test_thread.deleteLater)
        self.test_thread.start()

