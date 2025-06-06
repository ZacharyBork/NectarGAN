from typing import Literal

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
        self.last_epoch: int=0

        self._get_widgets()

    ### INIT HELPERS ###

    def _get_widgets(self) -> None:
        '''Get widgets accessed frequently by this class.'''
        self.statusbar = self.mainwidget.findChild(QStatusBar, 'statusbar')
        self.train_settings_grp = self.mainwidget.findChild(QFrame, 'train_settings')

        self.perf_graph = self.mainwidget.findChild(Graph, 'performance_graph')
        self.loss_g_graph = self.mainwidget.findChild(Graph, 'loss_g_graph')
        self.loss_d_graph = self.mainwidget.findChild(Graph, 'loss_d_graph')
        
        self.progress_bar_train = self.mainwidget.findChild(QProgressBar, 'train_progress')
        self.progress_bar_epoch = self.mainwidget.findChild(QProgressBar, 'epoch_progress')
        self.current_epoch_display = self.mainwidget.findChild(QLCDNumber, 'current_epoch')

        self.train_start_btn = self.mainwidget.findChild(QPushButton, 'train_start')
        self.train_stop_btn = self.mainwidget.findChild(QPushButton, 'train_stop')
        self.train_pause_btn = self.mainwidget.findChild(QPushButton, 'train_pause')

    ### SEND SIGNALS ###

    def _change_update_frequency(self, value: int) -> None:
        '''Sends a signal to the worker to change its update signal frequency.
        '''
        if hasattr(self, 'worker'):
            self.worker.change_update_frequency(value)

    ### RECEIVE SIGNALS ###

    def trainer_waiting(self, pulse: int) -> None:
        '''Recieves and processes a trainer waiting signal.
        
        When the trainer worker is paused, it will send a pulse signal every
        .5 seconds. This function takes that signal and uses it for a fun
        little waiting animation.

        Args:
            pulse : Signal from the worker.
        '''
        message = 'Waiting' + '.' * (pulse % 4)
        self.statusbar.showMessage(message)

    def _update_log(self, log_entry: str) -> None:
        '''Updates output log with training data.
        
        Args:
            log_entry : The entry to append to the log.
        '''
        if not self.log.output_frozen:
            self.log.write_entry(log_entry)
            if self.mainwidget.findChild(QCheckBox, 'autoscroll_log').isChecked():
                scroll = self.log.log_widget.verticalScrollBar()
                scroll.setValue(scroll.maximum())

    def _report_epoch_progress(self, progress: float) -> None:
        '''Reports training progress through the current epoch.
        
        Args:
            progress : [0, 1] progress through the current epoch.
        '''
        self.current_epoch_progress = progress
        self.progress_bar_epoch.setValue(progress * 100.0)

    def _report_train_progress(self, progress: tuple[int, float]) -> None:
        '''Reports epoch index and time at the end of each epoch.
        
        Args:
            progress : The int value is the index of the just completed epoch.
                The float value is the time, in seconds, which that epoch took
                to complete. 
        '''
        gen_lr = self.worker.config.train.generator.learning_rate
        total_epochs = gen_lr.epochs + gen_lr.epochs_decay
        
        self.last_epoch = progress[0] + 1
        self.train_progress = self.last_epoch / total_epochs
        
        self.progress_bar_train.setValue(int(self.train_progress*100.0))
        self.current_epoch_display.display(self.last_epoch+1)

        self.perf_graph.set_step(self.last_epoch)
        self.perf_graph.update_plot('epoch_time', progress[1])

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

    ### LOSSES ###

    def _get_losses(self, losses: dict[str, float]) -> None:
        loss_g_graph = self.mainwidget.findChild(Graph, 'loss_g_graph')
        loss_d_graph = self.mainwidget.findChild(Graph, 'loss_d_graph')
        losses_g = ['G_GAN', 'G_L1', 'G_SOBEL', 'G_LAP', 'G_VGG']
        losses_d = ['D_real', 'D_fake']

        step = 1.0 + self.current_epoch_progress + float(self.last_epoch)
        loss_g_graph.set_step(step)
        loss_d_graph.set_step(step)

        for name in losses_g:
            loss_g_graph.update_plot(name, losses[name])
            
        for name in losses_d:
            loss_d_graph.update_plot(name, losses[name])

    ### UI STATES ###
    def _set_ui_state(
            self, 
            state: Literal['train_start', 'train_paused', 'train_end']
        ) -> None:
        '''Sets various things in the main interface based on worker state.
        
        The function inits some UI values and sets train control button
        visibility. It also disables the training settings panel on train_start
        and re-enables it on train_end.

        Args:
            state : The current state, decides what to set the values to.
        '''
        state_values = {
            'train_start': [
                True, True, 0, 0, 1, False, False, 'Beginning Training'], 
            'train_paused': [], 
            'train_end': [
                False, False, 0, 0, 0, True, True, 'Training Complete']
        }
        def _set_state(x: list):
            self.train_start_btn.setHidden(x[0])
            self.train_stop_btn.setEnabled(x[1])
            self.progress_bar_train.setValue(x[2])
            self.progress_bar_epoch.setValue(x[3])
            self.current_epoch_display.display(x[4])
            self.train_settings_grp.setEnabled(x[5])
            self.train_pause_btn.setHidden(x[6])
        match state:
            case 'train_start' | 'train_end': _set_state(state_values[state])
            case 'train_paused':
                raise NotImplementedError(f'State {state} not implemented.')        

    ### TRAIN ###

    def start_train(self) -> None:
        '''Creates a QThread and starts a pix2pix training loop inside of it. 
        '''
        self.loss_g_graph.reset_graph()
        self.loss_d_graph.reset_graph()
        self.perf_graph.reset_graph()
        self.perf_graph.set_step(0.0)
        self.perf_graph.update_plot('epoch_time', 0.0)
        self.confighelper._build_launch_config()
        
        t = self.train_thread = QThread()
        w = self.worker = TrainerWorker(config=self.confighelper.config)
        self.worker.moveToThread(self.train_thread)

        self.safe_cleanup.connect(lambda : self._set_ui_state(state='train_end'))

        self.train_thread.started.connect(self.worker.run) # Training slot fn
        
        w.epoch_progress.connect(self._report_epoch_progress)
        w.train_progress.connect(self._report_train_progress)
        w.tensors.connect(self._get_tensors)     # XYZ image tensors
        w.losses.connect(self._get_losses)     # XYZ image tensors
        w.log.connect(self._update_log)          # Log strings
        w.waiting.connect(self.trainer_waiting)  # Trainer paused pulse
        
        w.finished.connect(self.train_thread.quit)  # Kill thread if finished,
        w.cancelled.connect(self.train_thread.quit) # or if canceled.
        w.finished.connect(self.safe_cleanup.emit)  # Also request safe cleanup
        w.cancelled.connect(self.safe_cleanup.emit) # for both signals.

        # deletelater on train_thread when train finishes
        t.finished.connect(self.train_thread.deleteLater)
        t.start() # Spin up train thread

        self._set_ui_state(state='train_start')

    def stop_train(self) -> None:
        '''Sends a signal to the worker telling it to stop training.'''
        if hasattr(self, 'worker'):
            self.worker.stop()
            self.statusbar.showMessage('Training Stopped', self.status_msg_length)

    def pause_train(self) -> None:
        '''Sends a signal to the worker telling it to pause training.'''
        if hasattr(self, 'worker'):
            if self.worker._is_paused:
                self.worker.unpause()
                self.statusbar.clearMessage()
                self.statusbar.showMessage('Training Resumed', self.status_msg_length)
                self.mainwidget.findChild(QPushButton, 'train_pause').setText('Pause Training')
            else:
                self.worker.pause()
                self.mainwidget.findChild(QPushButton, 'train_pause').setText('Resume Training')


