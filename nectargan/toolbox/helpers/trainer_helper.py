from typing import Literal
from pathlib import Path

import numpy as np
from torch import Tensor

from PySide6.QtCore import QObject, QThread, Signal, QVariantAnimation
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QLCDNumber, QFrame, QPushButton, QSpinBox, QLineEdit,
    QStatusBar, QCheckBox, QProgressBar, QLabel, QStackedWidget)

from nectargan.toolbox.helpers.config_helper import ConfigHelper
from nectargan.toolbox.components.log import OutputLog
from nectargan.toolbox.widgets.graph import Graph
from nectargan.toolbox.widgets.imagelabel import ImageLabel
from nectargan.toolbox.utils.signal_timer import SignalTimer
from nectargan.toolbox.workers.pix2pix_trainerworker import TrainerWorker
from nectargan.toolbox.components.settings_dock import SettingsDock
  
class TrainerHelper(QObject):
    ### SIGNALS ###
    safe_cleanup = Signal() # Worker shutdown signal

    def __init__(
            self, 
            mainwidget: QWidget,
            confighelper: ConfigHelper,
            settings_dock: SettingsDock,
            log: OutputLog,
            status_msg_length: int=2000
        ) -> None:
        super().__init__()
        self.mainwidget = mainwidget
        self.find = self.mainwidget.findChild

        self.confighelper = confighelper
        self.settings_dock = settings_dock
        self.log = log
        self.status_msg_length = status_msg_length
        
        self.worker: TrainerWorker | None = None
        self.pixmaps: list[QPixmap] = []
        self.last_epoch: int=0

        self.epoch_timer = SignalTimer()
        self.iter_timer = SignalTimer()
        self.train_timer = SignalTimer()

        self._get_widgets()
        self.warning_label = self.find(QLabel, 'training_warning_label')
        self.warning_label.setHidden(True)
        self.warning_anim: QVariantAnimation | None = None
        self._init_warning_animation()

        self.find(QCheckBox, 'log_losses').clicked.connect(
            lambda x : self.find(QFrame, 'log_loss_settings').setEnabled(x))
        self.find(QSpinBox, 'input_nc').setEnabled(False)

    ### INIT HELPERS ###

    def _get_widgets(self) -> None:
        '''Get widgets accessed frequently by this class.'''
        self.statusbar = self.find(QStatusBar, 'statusbar')
        self.train_settings_grp = self.find(QFrame, 'train_settings')

        self.perf_graph = self.find(Graph, 'performance_graph')
        self.loss_g_graph = self.find(Graph, 'loss_g_graph')
        self.loss_d_graph = self.find(Graph, 'loss_d_graph')
        
        self.progress_bar_train = self.find(QProgressBar, 'train_progress')
        self.progress_bar_epoch = self.find(QProgressBar, 'epoch_progress')
        self.current_epoch_display = self.find(QLCDNumber, 'current_epoch')

        self.train_start_btn = self.find(QPushButton, 'train_start')
        self.train_stop_btn = self.find(QPushButton, 'train_stop')
        self.train_pause_btn = self.find(QPushButton, 'train_pause')

    ### WARNING LABEL ###

    def _show_warning_label(self, val: int) -> None:
        show = not val > 0
        self.warning_label.setHidden(show)

    def _init_warning_animation(self) -> None:
        if not self.warning_anim is None: 
            self.warning_anim.deleteLater()
        self.warning_anim = QVariantAnimation()
        self.warning_anim.setStartValue(10000)
        self.warning_anim.setEndValue(0)
        self.warning_anim.setDuration(3000)
        self.warning_anim.valueChanged.connect(
            lambda x : self.warning_label.setHidden(not x > 0))

    def _warn(self, warning: str) -> None:
        self._init_warning_animation()
        self.warning_label.setText(warning)
        self.warning_anim.start()

    ### SEND SIGNALS ###

    def _change_update_frequency(self, value: int) -> None:
        '''Sends a signal to the worker to change its update signal frequency.
        '''
        if not self.worker is None:
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
            if self.find(QCheckBox, 'autoscroll_log').isChecked():
                scroll = self.log.log_widget.verticalScrollBar()
                scroll.setValue(scroll.maximum())

    def _report_epoch_progress(self, progress: float) -> None:
        '''Reports training progress through the current epoch.
        
        Args:
            progress : [0, 1] progress through the current epoch.
        '''
        self.iter_timer.set_time()
        self.find(QLabel, 'performance_iter_fastest').setText(
            self.iter_timer.get_time('fastest'))
        self.find(QLabel, 'performance_iter_slowest').setText(
            self.iter_timer.get_time('slowest'))
        self.find(QLabel, 'performance_iter_average').setText(
            self.iter_timer.get_time('average'))
        
        self.train_timer.set_time()
        self.find(QLabel, 'performance_time_total').setText(
            self.train_timer.get_time('total'))

        self.current_epoch_progress = progress
        self.progress_bar_epoch.setValue(progress * 100.0)

    def _report_train_progress(self, progress: tuple[int, float]) -> None:
        '''Reports epoch index and time at the end of each epoch.
        
        Args:
            progress : The int value is the index of the just completed epoch.
                The float value is the time, in seconds, which that epoch took
                to complete. 
        '''
        self.epoch_timer.set_time()
        self.find(QLabel, 'performance_epoch_fastest').setText(
            self.epoch_timer.get_time('fastest'))
        self.find(QLabel, 'performance_epoch_slowest').setText(
            self.epoch_timer.get_time('slowest'))
        self.find(QLabel, 'performance_epoch_average').setText(
            self.epoch_timer.get_time('average'))

        gen_lr = self.worker.config.train.generator.learning_rate
        total_epochs = gen_lr.epochs + gen_lr.epochs_decay
        
        self.last_epoch = progress[0] + 1
        self.train_progress = self.last_epoch / total_epochs
        
        self.progress_bar_train.setValue(int(self.train_progress*100.0))
        self.current_epoch_display.display(self.last_epoch+1)

        self.perf_graph.update_plot('epoch_time', progress[1], self.last_epoch)

    ### TRAIN PROGRESS IMAGES ###

    def _show_xyz_images(self) -> None:
        '''Updates XYZ image display with `self.pixmaps`.
        
        Called at the end of `_get_tensors()`, this function takes the result
        of the Tensor -> nparray -> QImage -> QPixmap pipeline in that function
        and displays it to the UI.
        '''
        self.find(ImageLabel, 'x_label').set_pixmap(self.pixmaps[0])
        self.find(ImageLabel, 'y_label').set_pixmap(self.pixmaps[1])
        self.find(ImageLabel, 'y_fake_label').set_pixmap(self.pixmaps[2])

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
            qimage = QImage(np_image.data, w, h, QImage.Format_RGB888)
            self.pixmaps.append(QPixmap.fromImage(qimage))
        self._show_xyz_images()

    ### LOSSES ###

    def _get_losses(self, losses: dict[str, float]) -> None:
        loss_g_graph = self.find(Graph, 'loss_g_graph')
        loss_d_graph = self.find(Graph, 'loss_d_graph')
        losses_g = ['G_GAN', 'G_L1', 'G_L2', 'G_SOBEL', 'G_LAP', 'G_VGG']
        losses_d = ['D_real', 'D_fake']

        step = 1.0 + self.current_epoch_progress + float(self.last_epoch)
        for name in losses_g: 
            loss_g_graph.update_plot(name, losses[name], step)
        for name in losses_d: 
            loss_d_graph.update_plot(name, losses[name], step)

    ### UI STATES ###
    def _set_ui_state(
            self, 
            state: Literal['train_start', 'train_paused', 'train_end']
        ) -> None:
        '''Sets various things in the main interface based on worker state.

        Args:
            state : The current state, decides what to set the values to.
        '''
        match state:
            case 'train_start':
                self.train_start_btn.setHidden(True)
                self.train_stop_btn.setHidden(False)
                self.train_pause_btn.setHidden(False)
                self.progress_bar_train.setValue(0)
                self.progress_bar_epoch.setValue(0)
                self.current_epoch_display.display(1)
            case 'train_end':
                self.train_start_btn.setHidden(False)
                self.train_stop_btn.setHidden(True)
                self.train_pause_btn.setHidden(True)
                self.progress_bar_train.setValue(0)
                self.progress_bar_epoch.setValue(0)
                self.current_epoch_display.display(0)
            case 'train_paused':
                raise NotImplementedError(f'State {state} not implemented.')        

    ### TRAIN ###

    def _init_timers(self) -> None:
        '''Builds new timers for training session tracking.'''
        self.epoch_timer = SignalTimer()
        self.iter_timer = SignalTimer()
        self.train_timer = SignalTimer()

    def _init_graphs(self) -> None:
        '''Resets and initializes training related UI graphs.'''
        self.loss_g_graph.reset_graph()
        self.loss_d_graph.reset_graph()
        self.perf_graph.reset_graph()
        self.perf_graph.update_plot('epoch_time', 0.0, 0.0)

    def _connect_worker_signals(self) -> None:
        '''Hooks up signals for the trainer worker.'''
        w = self.worker
        
        w.epoch_progress.connect(self._report_epoch_progress)
        w.train_progress.connect(self._report_train_progress)
        
        w.tensors.connect(self._get_tensors)        # XYZ image tensors
        w.losses.connect(self._get_losses)          # Loss values
        w.log.connect(self._update_log)             # Log strings
        
        w.waiting.connect(self.trainer_waiting)     # Trainer paused pulse
        w.finished.connect(self.train_thread.quit)  # Kill thread if finished,
        w.cancelled.connect(self.train_thread.quit) # or if canceled.
        w.finished.connect(self.safe_cleanup.emit)  # Also request safe cleanup
        w.cancelled.connect(self.safe_cleanup.emit) # for both signals.

    def _preflight_check(self) -> bool:
        output_directory = self.find(QLineEdit, 'output_directory').text()
        if not output_directory == '':
            output_directory = Path(output_directory)
            if not output_directory.exists():
                self._warn('Please enter a valid Output Root to continue.')
                return False
        else: 
            self._warn('Please enter a valid Output Root to continue.')
            return False
        experiment_name = self.find(QLineEdit, 'experiment_name').text() 
        if experiment_name == '':
            self._warn('Please enter a valid Experiment Name to continue.')
            return False
        dataroot = self.find(QLineEdit, 'dataroot').text()
        if not dataroot == '':
            dataroot = Path(dataroot)
            if not dataroot.exists():
                self._warn('Please enter a valid Dataset Root to continue.')
                return False
            if not Path(dataroot, 'train').exists(): 
                self._warn(
                    f'Unable to locate training dataset files.\n'
                    f'Please ensure the dataset path is correct.')
                return False
        else: 
            self._warn('Please enter a valid Dataset Root to continue.') 
            return False 
        return True

    def start_train(self) -> None:
        '''Creates a QThread and starts a pix2pix training loop inside of it. 
        '''
        success = self._preflight_check()
        if not success: return
        self.last_epoch = 0               # Reset both progress
        self.current_epoch_progress = 0.0 # tracking variables        

        # Build JSON-like config data from UI settings
        self.confighelper._build_launch_config() 
        
        # Start thread, create worker, move worker to new thread
        self.train_thread = QThread() 
        try:
            self.worker = TrainerWorker(
                config=self.confighelper.config,
                log_losses=self.find(QCheckBox, 'log_losses').isChecked(),
                loss_dump_frequency=self.find(
                    QSpinBox, 'loss_dump_frequency').value())
        except ValueError as e: return self._warn(str(e))
        self.worker.moveToThread(self.train_thread)

        self.safe_cleanup.connect(self.stop_train) # Hook up helper signals
        self.train_thread.started.connect(self.worker.run)
        
        self._connect_worker_signals() # Then worker signals
        
        # deletelater on train_thread when train finishes
        self.train_thread.finished.connect(self.train_thread.deleteLater)
        self.train_thread.start()   # Spin up train thread

        self.settings_dock.set_state('training')

        self._set_ui_state(state='train_start')
        self.find(QStackedWidget, 'train_page_state_swap').setCurrentIndex(1)

        self._init_timers() # Create new timers for this train
        self._init_graphs() # And delete all previous graph data
        self.epoch_timer.set_time() # Start epoch timer
        self.iter_timer.set_time()  # Start iteration timer
        self.train_timer.set_time() # Start train timer
        

    def _cleanup(self) -> None:
        '''Cleanup when training is finished.'''
        self._set_ui_state(state='train_end')
        self.settings_dock.set_state('init')
        self.find( # Reset train settings page
            QStackedWidget, 'train_page_state_swap').setCurrentIndex(0)
        self.statusbar.showMessage(
            'Training Stopped', self.status_msg_length)

    def stop_train(self) -> None:
        '''Sends a signal to the worker telling it to stop training.'''
        if not hasattr(self, 'worker'): return
        self.worker.stop()
        self._cleanup()

    def pause_train(self) -> None:
        '''Sends a signal to the worker telling it to pause training.'''
        if not hasattr(self, 'worker'): return
        
        if self.worker._is_paused:
            self.worker.unpause()
            self.statusbar.clearMessage()
            self.statusbar.showMessage(
                'Training Resumed', self.status_msg_length)
            self.find(QPushButton, 'train_pause').setText('Pause Training')
        else:
            self.worker.pause()
            self.find(QPushButton, 'train_pause').setText('Resume Training')


