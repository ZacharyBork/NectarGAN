from torch import Tensor
import numpy as np

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QLCDNumber, QFrame, QPushButton, 
    QStatusBar, QCheckBox, QProgressBar
)

from pix2pix_graphical.ui.utils.config_helper import ConfigHelper
from pix2pix_graphical.ui.utils.log import OutputLog
from pix2pix_graphical.ui.utils.graphing import Graph
from pix2pix_graphical.ui.utils.imagelabel import ImageLabel
from pix2pix_graphical.ui.workers.pix2pix_trainerworker import TrainerWorker

class TrainerHelper(QObject):
    ### SIGNALS ###
    safe_cleanup = Signal() # Worker shutdown signal

    def __init__(
            self, 
            mainwidget: QWidget,
            confighelper: ConfigHelper,
            log: OutputLog,
            status_msg_length: int=2000
        ) -> None:
        super().__init__()
        self.mainwidget = mainwidget
        self.confighelper = confighelper
        self.log = log
        self.status_msg_length = status_msg_length

        self.worker: TrainerWorker | None = None
        self.pixmaps: list[QPixmap] = []

        self._get_widgets()

    ### INIT HELPERS ###

    def _get_widgets(self) -> None:
        self.statusbar = self.mainwidget.findChild(QStatusBar, 'statusbar')
        self.train_settings_grp = self.mainwidget.findChild(QFrame, 'train_settings')
        self.perf_graph = self.mainwidget.findChild(Graph, 'performance_graph')
        
        self.progress_bar_train = self.mainwidget.findChild(QProgressBar, 'train_progress')
        self.progress_bar_epoch = self.mainwidget.findChild(QProgressBar, 'epoch_progress')
        self.current_epoch_display = self.mainwidget.findChild(QLCDNumber, 'current_epoch')

        self.train_start_btn = self.mainwidget.findChild(QPushButton, 'train_start')
        self.train_stop_btn = self.mainwidget.findChild(QPushButton, 'train_stop')
        self.train_pause_btn = self.mainwidget.findChild(QPushButton, 'train_pause')

    ### SIGNAL HANDLERS ###

    def change_update_frequency(self, value: int) -> None:
        if hasattr(self, 'worker'):
            self.worker.change_update_frequency(value)

    def update_log(self, log_entry: str) -> None:
        if not self.log.output_frozen:
            self.log.write_entry(log_entry)
            if self.mainwidget.findChild(QCheckBox, 'autoscroll_log').isChecked():
                scroll = self.log.log_widget.verticalScrollBar()
                scroll.setValue(scroll.maximum())

    def report_epoch_progress(self, value: int) -> None:
        self.progress_bar_epoch.setValue(value)

    def report_train_progress(self, progress: tuple[int, float]) -> None:
        gen_lr = self.worker.config.train.generator.learning_rate
        total_epochs = gen_lr.epochs + gen_lr.epochs_decay
        
        self.last_epoch = progress[0] + 1
        self.train_progress = self.last_epoch / total_epochs
        
        self.progress_bar_train.setValue(int(self.train_progress*100.0))
        self.current_epoch_display.display(self.last_epoch+1)
        self.perf_graph.update_graph(self.last_epoch, progress[1])

    ### TRAIN PROGRESS IMAGES ###

    def _show_xyz_images(self) -> None:
        '''Updates XYZ image display with `self.pixmaps`.
        
        Called at the end of `_get_tensors()`, this function takes the result
        of the Tensor -> nparray -> QImage -> QPixmap pipeline in that function
        and displays it to the UI.
        '''
        self.mainwidget.findChild(ImageLabel, 'x_label').set_pixmap(self.pixmaps[0])
        self.mainwidget.findChild(ImageLabel, 'y_label').set_pixmap(self.pixmaps[1])
        self.mainwidget.findChild(ImageLabel, 'y_fake_label').set_pixmap(self.pixmaps[2])

    def _get_tensors(self, tensors: tuple[Tensor]) -> None:
        '''Signal handler for retrieving and processing tensors from worker.

                  Remap [0, 255]
                       ⌄⌄
        `Tensor` -> `nparray` -> `QImage` -> `QPixmap` -> `_show_xyz_images()`
                 ^^
           Squeeze/Denorm
        '''
        self.pixmaps = [] # Reset stored pixmaps
        # Denorm Tanh output tensor: [-1, 1] -> [0, 1]
        denorm = lambda x : x * 0.5 + 0.5
        for tensor in tensors:
            tensor = denorm(tensor.squeeze(0))
            np_image = tensor.numpy()
            np_image = np.transpose(np_image, (1, 2, 0))
            np_image = (np_image * 255).astype(np.uint8)

            h, w = np_image.shape[:2]
            np_image = np.ascontiguousarray(np_image)
            qimage = QImage(
                np_image.data, w, h, 
                QImage.Format_RGB888)
            self.pixmaps.append(QPixmap.fromImage(qimage))
        self._show_xyz_images()

    ### TRAIN ###

    def start_train(self) -> None:
        '''Creates a QThread and starts a pix2pix training loop inside of it. 
        '''
        self.perf_graph.reset_graph()
        self.perf_graph.update_graph(0.0, 0.0)
        self.confighelper._build_launch_config()
        
        self.train_thread = QThread()
        self.worker = TrainerWorker(config=self.confighelper.config)
        self.worker.moveToThread(self.train_thread)

        self.safe_cleanup.connect(self.cleanup)

        self.train_thread.started.connect(self.worker.run) # Training slot fn
        
        self.worker.epoch_progress.connect(self.report_epoch_progress)
        self.worker.train_progress.connect(self.report_train_progress)
        self.worker.tensors.connect(self._get_tensors)     # XYZ image tensors
        self.worker.log.connect(self.update_log)           # Log strings
        self.worker.waiting.connect(self.trainer_waiting)
        
        self.worker.finished.connect(self.train_thread.quit)  # Cancel Handler
        self.worker.cancelled.connect(self.train_thread.quit) # Cancel Handler

        self.train_thread.finished.connect(self.train_thread.deleteLater)
        self.worker.finished.connect(self.safe_cleanup.emit)
        self.worker.cancelled.connect(self.safe_cleanup.emit)

        self.train_thread.start()
        self.train_start_btn.setHidden(True)
        self.train_stop_btn.setEnabled(True)
        self.current_epoch_display.display(1)
        self.train_settings_grp.setEnabled(False)
        self.mainwidget.findChild(QPushButton, 'train_pause').setHidden(False)

        self.statusbar.showMessage('Beginning Training...', self.status_msg_length)

    def stop_train(self) -> None:
        if hasattr(self, 'worker'):
            self.worker.stop()
            self.statusbar.showMessage('Training Stopped', self.status_msg_length)

    def pause_train(self) -> None:
        if hasattr(self, 'worker'):
            if self.worker._is_paused:
                self.worker.unpause()
                self.statusbar.clearMessage()
                self.statusbar.showMessage('Training Resumed', self.status_msg_length)
                self.mainwidget.findChild(QPushButton, 'train_pause').setText('Pause Training')
            else:
                self.worker.pause()
                self.mainwidget.findChild(QPushButton, 'train_pause').setText('Resume Training')

    def trainer_waiting(self, counter: int) -> None:
        message = 'Waiting' + '.' * (counter % 4)
        self.statusbar.showMessage(message)

    ### CLEANUP ###
    
    def cleanup(self):
        self.train_start_btn.setHidden(False)
        self.train_stop_btn.setEnabled(False)
        self.progress_bar_train.setValue(0)
        self.progress_bar_epoch.setValue(0)
        self.current_epoch_display.display(0)
        self.train_settings_grp.setEnabled(True)
        self.train_pause_btn.setHidden(True)
