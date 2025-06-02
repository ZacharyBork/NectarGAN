import sys
import pathlib
import json
from typing import Union

import PySide6.QtWidgets as QtWidgets
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QThread, QObject, Signal

from pix2pix_graphical.ui.workers import TrainerWorker

class Interface(QObject):
    safe_cleanup_requested = Signal()

    def __init__(self):
        super().__init__()

    def start_train(self):
        self.train_thread = QThread()
        self.worker = TrainerWorker()
        self.worker.moveToThread(self.train_thread)

        self.train_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.report_progress)
        self.worker.finished.connect(self.train_thread.quit)
        self.worker.cancelled.connect(self.train_thread.quit)

        self.train_thread.finished.connect(self.train_thread.deleteLater)
        self.worker.finished.connect(self.safe_cleanup_requested.emit)
        self.worker.cancelled.connect(self.safe_cleanup_requested.emit)

        self.train_thread.start()
        self.mainwidget.findChild(QtWidgets.QPushButton, 'train_start').setEnabled(False)
        self.mainwidget.findChild(QtWidgets.QPushButton, 'train_stop').setEnabled(True)

    def stop_train(self):
        if hasattr(self, 'worker'):
            self.worker.stop()

    def report_progress(self, value):
        self.mainwidget.findChild(QtWidgets.QProgressBar, 'train_progress').setValue(value)

    def cleanup(self):
        self.mainwidget.findChild(QtWidgets.QPushButton, 'train_start').setEnabled(True)
        self.mainwidget.findChild(QtWidgets.QPushButton, 'train_stop').setEnabled(False)
        self.mainwidget.findChild(QtWidgets.QProgressBar, 'train_progress').setValue(0)

    def update_spinbox_from_slider(
            self,
            spinbox: Union[QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox], 
            slider: QtWidgets.QSlider, multiplier: float=1.0):
        spinbox.setValue(float(slider.value()) * multiplier)

    # Assign callback functions

    def link_sliders_to_spinboxes(self):
        sliders_to_link = [
            # (slider, spinbox, multiplier)
            ('lr_slider', 'learning_rate', 0.0001),
            ('num_epochs_slider', 'num_epochs', 1.0),
            ('batch_size_slider', 'batch_size', 1.0),
            ('worker_count_slider', 'worker_count', 1.0),
            ('beta1_slider', 'beta1', 0.01),
            ('l1_slider', 'l1', 1.0),
            ('sobel_slider', 'sobel', 1.0)
        ]

        for x, y, z in sliders_to_link:
            slider = self.mainwidget.findChild(QtWidgets.QSlider, x)
            spinbox = self.mainwidget.findChild(QtWidgets.QSpinBox, y)
            if spinbox == None: spinbox = self.mainwidget.findChild(QtWidgets.QDoubleSpinBox, y)
            slider.valueChanged.connect(
                lambda value, s=spinbox, sl=slider, m=z: self.update_spinbox_from_slider(s, sl, multiplier=m))

    def init_ui(self):
        self.link_sliders_to_spinboxes()

        self.mainwidget.findChild(QtWidgets.QProgressBar, 'train_progress').setValue(0)
        self.mainwidget.findChild(QtWidgets.QPushButton, 'train_start').clicked.connect(self.start_train)
        self.mainwidget.findChild(QtWidgets.QPushButton, 'train_stop').clicked.connect(self.stop_train)
        self.mainwidget.findChild(QtWidgets.QPushButton, 'train_stop').setEnabled(False)

        # test_button = self.mainwidget.findChild(QtWidgets.QPushButton, 'test_button')
        # test_button.clicked.connect(start_training)

        # coninue_train = self.mainwidget.findChild(QtWidgets.QCheckBox, 'continue_train')
        # load_epoch = self.mainwidget.findChild(QtWidgets.QSpinBox, 'load_epoch')


        self.safe_cleanup_requested.connect(self.cleanup)

    def run(self):
        app = QtWidgets.QApplication(sys.argv)

        loader = QUiLoader()
        
        root = pathlib.Path(__file__).parent
        ui_file = pathlib.Path(root, 'resources', 'base.ui')
        file = QFile(ui_file.resolve().as_posix())
        file.open(QFile.ReadOnly)
        
        self.mainwidget = loader.load(file)
        file.close()
        self.init_ui()

        self.mainwidget.show()
        sys.exit(app.exec())

