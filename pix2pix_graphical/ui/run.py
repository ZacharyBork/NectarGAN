import sys
import json
import pathlib
from typing import Union, Any
from importlib.resources import files

import torch
import numpy as np
# import pyqtgraph as qtg
import PySide6.QtWidgets as QtWidgets
import PySide6.QtGui as QtGui
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import Qt, QFile, QThread, QObject, Signal, QEvent

from pix2pix_graphical.ui.workers import TrainerWorker
from pix2pix_graphical.ui.config_helper import ConfigHelper

class ImageLabel(QtWidgets.QLabel):
    def __init__(
            self, 
            name: str,
            pixmap: QtGui.QPixmap | None=None,
            parent: QtWidgets.QWidget=None
        ) -> None:
        super().__init__(parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding)
        
        self._pixmap: QtGui.QPixmap = pixmap
        if not pixmap is None: self._rescale()
        self._name: str = name
        
    def set_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        self._pixmap = pixmap
        self._rescale()

    def resizeEvent(self, event: QEvent) -> None:
        self._rescale()
        super().resizeEvent(event)

    def _rescale(self) -> None:
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled)

class Interface(QObject):
    ### INTERFACE SIGNALS ###
    safe_cleanup = Signal() # Worker shutdown signal
    
    def __init__(self) -> None:
        '''Init function for the `Interface` class.'''
        super().__init__()
        self.config_helper = ConfigHelper()

        # Stores pixmaps converted from Tensors returned by worker
        self.pixmaps: list[QtGui.QPixmap] = []

        # Stores references to frequently used widgets
        self.widgets: dict[str, QtWidgets.QWidget] = {}

    ### CONFIG HANDLING ###

    def _get_config_data(
            self, 
            path: pathlib.Path | None=None
        ) -> dict[str, Any]:
        if path is None:
            path = files('pix2pix_graphical.config').joinpath('default.json')
        else: path = path.as_posix()
        try:
            with open(path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            message = f'Unable to open config file at: {path.as_posix()}'
            raise RuntimeError(message) from e

    def _init_from_config(self, config_path: pathlib.Path | None=None) -> None:
        config_data = self._get_config_data(path=config_path)['config']
        
        for ui_name, info in self.config_helper.CONFIG_MAP.items():
            config_keys, widget_type = info
            keys = config_keys.split('.')
            value = config_data
            for key in keys:
                value = value[key]

            widget = self._get(widget_type, ui_name)
            self.config_helper.WIDGET_SETTERS[widget_type](widget, value)
            
    ### TRAINING ###

    def start_train(self) -> None:
        '''Creates a QThread and starts a pix2pix training loop inside of it. 
        '''
        self.train_thread = QThread()
        self.worker = TrainerWorker()
        self.worker.moveToThread(self.train_thread)

        self.train_thread.started.connect(self.worker.run) # Training slot fn
        
        self.worker.epoch_progress.connect(self.report_epoch_progress)
        self.worker.train_progress.connect(self.report_train_progress)
        self.worker.tensors.connect(self._get_tensors)     # XYZ image tensors
        self.worker.log.connect(self.update_log)           # Log strings
        
        self.worker.finished.connect(self.train_thread.quit)  # Cancel Handler
        self.worker.cancelled.connect(self.train_thread.quit) # Cancel Handler

        self.train_thread.finished.connect(self.train_thread.deleteLater)
        self.worker.finished.connect(self.safe_cleanup.emit)
        self.worker.cancelled.connect(self.safe_cleanup.emit)

        self.train_thread.start()
        self.widgets['train_start'].setEnabled(False)
        self.widgets['train_stop'].setEnabled(True)
        self._get(QtWidgets.QLCDNumber, 'current_epoch').display(1)
        self._get(QtWidgets.QFrame, 'train_settings').setEnabled(False)

    ### GENERAL SIGNAL HANDLERS

    def resizeEvent(self, event: QtGui.QResizeEvent):
        # self._show_xyz_images()
        super().resizeEvent(event)

    def stop_train(self) -> None:
        if hasattr(self, 'worker'):
            self.worker.stop()

    def change_update_frequency(self, value: int) -> None:
        if hasattr(self, 'worker'):
            self.worker.change_update_frequency(value)

    def update_log(self, log_entry: str) -> None:
        self.widgets['output_log'].append(log_entry)
        if self._get(QtWidgets.QCheckBox, 'autoscroll_log').isChecked():
            scroll = self.widgets['output_log'].verticalScrollBar()
            scroll.setValue(scroll.maximum())

    def report_epoch_progress(self, value: int) -> None:
        self.widgets['epoch_progress'].setValue(value)

    def report_train_progress(self, value: int) -> None:
        gen_lr = self.worker.config.train.generator.learning_rate
        total_epochs = gen_lr.epochs + gen_lr.epochs_decay
        self.widgets['train_progress'].setValue(
            int(((value+1) / total_epochs) * 100.0))
        self._get(QtWidgets.QLCDNumber, 'current_epoch').display(value+2)

    ### UPDATE IMAGES ###

    def _show_xyz_images(self) -> None:
        '''Updates XYZ image display with `self.pixmaps`.
        
        Called at the end of `_get_tensors()`, this function takes the result
        of the Tensor -> nparray -> QImage -> QPixmap pipeline in that function
        and displays it to the UI.
        '''
        self.widgets['x_label'].set_pixmap(self.pixmaps[0])
        self.widgets['y_label'].set_pixmap(self.pixmaps[1])
        self.widgets['y_fake_label'].set_pixmap(self.pixmaps[2])

    def _get_tensors(self, tensors: tuple[torch.Tensor]) -> None:
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
            qimage = QtGui.QImage(
                np_image.data, w, h, 
                QtGui.QImage.Format_RGB888)
            self.pixmaps.append(QtGui.QPixmap.fromImage(qimage))
        self._show_xyz_images()

    ### CLEANUP ###

    def cleanup(self):
        self.widgets['train_start'].setEnabled(True)
        self.widgets['train_stop'].setEnabled(False)
        self.widgets['train_progress'].setValue(0)
        self.widgets['epoch_progress'].setValue(0)
        self._get(QtWidgets.QFrame, 'train_settings').setEnabled(True)

    ### INIT HELPERS ###

    def _get(
            self, 
            type: QtWidgets.QWidget,
            name: str
        ) -> QtWidgets.QWidget:
        '''Gets a child widget of self.mainwidget by type and name.

        This wrapper function exists is so that I don't need to write 
        self.mainwidget.findChild all over the place. This isn't much better
        but it's not quite as awful to look at.

        Args:
            type : The type of the QWidget you want to access.
            name : The name of the QWidget you want to access.

        Returns:
            QWidget : The requested widget.
        '''
        return self.mainwidget.findChild(type, name)

    def _set_button_icon(
            self, 
            filename: str, 
            button: QtWidgets.QPushButton
        ) -> None:
        root = pathlib.Path(__file__).parent
        file = pathlib.Path(root, 'resources', 'icons', filename)
        icon = QtGui.QIcon(file.as_posix())
        button.setIcon(icon)

    def _update_spinbox_from_slider(
            self,
            spinbox: Union[QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox], 
            slider: QtWidgets.QSlider, 
            multiplier: float=1.0
        ) -> None:
        '''Sets the value of a QSpinbox-type widget to the value of a QSlider.
        '''
        spinbox.setValue(float(slider.value()) * multiplier)

    def _sliders_to_spinboxes(self) -> None:
        '''Links all spinboxes in the interface to their respective sliders.'''
        sliders_to_link = [
            # (slider, spinbox, multiplier)
            ('update_frequency_slider', 'update_frequency', 1.0),

            ('batch_size_slider', 'batch_size', 1.0),
            ('gen_lr_initial_slider', 'gen_lr_initial', 0.0001),
            ('gen_lr_target_slider', 'gen_lr_target', 0.0001),
            ('gen_optim_beta1_slider', 'gen_optim_beta1', 0.01),
            
            ('disc_lr_initial_slider', 'disc_lr_initial', 0.0001),
            ('disc_lr_target_slider', 'disc_lr_target', 0.0001),
            ('disc_optim_beta1_slider', 'disc_optim_beta1', 0.01),
            
            ('lambda_l1_slider', 'lambda_l1', 1.0),
            ('lambda_sobel_slider', 'lambda_sobel', 1.0),
            ('lambda_laplacian_slider', 'lambda_laplacian', 1.0),
            ('lambda_vgg_slider', 'lambda_vgg', 1.0)]

        for x, y, z in sliders_to_link:
            slider = self.mainwidget.findChild(QtWidgets.QSlider, x)
            spinbox = self.mainwidget.findChild(QtWidgets.QSpinBox, y)
            if spinbox == None: spinbox = self.mainwidget.findChild(QtWidgets.QDoubleSpinBox, y)
            slider.valueChanged.connect(
                lambda value, s=spinbox, sl=slider, m=z: self._update_spinbox_from_slider(s, sl, multiplier=m))

    ### CALLBACKS ###

    def close_experiment_settings(self) -> None:
        dock = self._get(QtWidgets.QDockWidget, 'experiment_settings')
        button = self._get(QtWidgets.QPushButton, 'close_experiment_settings')
        direction = 'left' if dock.isHidden() else 'right'
        self._set_button_icon(f'caret-line-{direction}-bold.svg', button)
        dock.setHidden(not dock.isHidden())

    def _split_lr_schedules(self) -> None:
        gen_box = self._get(QtWidgets.QGroupBox, 'gen_schedule_box')
        disc_box = self._get(QtWidgets.QGroupBox, 'disc_schedule_box')
        checkbox = self._get(QtWidgets.QCheckBox, 'separate_lr_schedules')
        if checkbox.isChecked():
            gen_box.setTitle('Generator')
            disc_box.setHidden(False)
        else:
            gen_box.setTitle('Schedule')
            disc_box.setHidden(True)

    ### INIT INTERFACE ###

    def _get_signal_widgets(self) -> None:
        self.widgets = {
            'output_log': self._get(QtWidgets.QTextEdit, 'output_log'),
            'train_start': self._get(QtWidgets.QPushButton, 'train_start'),
            'train_stop': self._get(QtWidgets.QPushButton, 'train_stop'),
            'epoch_progress': self._get(QtWidgets.QProgressBar, 'epoch_progress'),
            'train_progress': self._get(QtWidgets.QProgressBar, 'train_progress'),
            'x_label': self._get(QtWidgets.QLabel, 'x_label'),
            'y_label': self._get(QtWidgets.QLabel, 'y_label'),
            'y_fake_label': self._get(QtWidgets.QLabel, 'y_fake_label')}

    def _init_output_log(self) -> None:
        log = self.widgets['output_log']
        log.setReadOnly(True)
        log.viewport().setCursor(QtGui.QCursor(Qt.CursorShape.ArrowCursor))
        log.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        log.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)

    def _init_training_controls(self) -> None:
        self.widgets['train_progress'].setValue(0)
        self.widgets['epoch_progress'].setValue(0)
        self.widgets['train_start'].clicked.connect(self.start_train)
        self.widgets['train_stop'].clicked.connect(self.stop_train)
        self.widgets['train_stop'].setEnabled(False)

    def _swap_image_labels(self) -> None:
        for name in ['x_label', 'y_label', 'y_fake_label']:
            old_label = self.widgets[name]
            layout = old_label.parentWidget().layout()

            new_label = ImageLabel(name=name, pixmap=None, parent=self.mainwidget)
            new_label.setObjectName(name)
            new_label.setMinimumSize(old_label.minimumSize())
            new_label.setSizePolicy(old_label.sizePolicy())
            new_label.setAlignment(old_label.alignment()) 

            layout.replaceWidget(old_label, new_label)
            old_label.deleteLater()
            self.widgets[name] = new_label

    def _init_pushbuttons(self) -> None:
        _button = QtWidgets.QPushButton

        button = self._get(_button, 'close_experiment_settings')
        self._set_button_icon('caret-line-left-bold.svg', button)
        button.clicked.connect(self.close_experiment_settings)

    def init_ui(self) -> None:
        '''Initialized the interface. Links parameters, sets defaults, etc.'''
        self._get_signal_widgets()     # Get commonly used widgets
        self._init_output_log()        # Set some overrides on the output log
        self._sliders_to_spinboxes()   # Link sliders to their spinboxes
        self._init_training_controls() # Link callbacks for training controls
        self._swap_image_labels()      # Replace QLabels with custom wrappers
        self._init_pushbuttons()

        

        self._get(QtWidgets.QGroupBox, 'disc_schedule_box').setHidden(True)
        self._get(QtWidgets.QCheckBox, 'separate_lr_schedules').clicked.connect(self._split_lr_schedules)
        update_freq = self._get(QtWidgets.QSpinBox, 'update_frequency')
        update_freq.valueChanged.connect(lambda x=update_freq.value() : self.change_update_frequency(x))

        current_epoch = self._get(QtWidgets.QLCDNumber, 'current_epoch')
        current_epoch.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
        palette = current_epoch.palette()
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor('black'))
        current_epoch.setPalette(palette)


        self._get(QtWidgets.QPushButton, 'test').clicked.connect(lambda : self._init_from_config(config_path=None))


        self.safe_cleanup.connect(self.cleanup)

    ### ENTRYPOINT ###

    def _get_ui_file(self) -> QFile:
        '''Gets the main widget's UI file and returns it as a QFile.
        
        Returns:
            QFile : The UI file.
        
        Raises:
            FileNotFoundError : If unable to locate UI file.
        '''
        root = pathlib.Path(__file__).parent
        file = pathlib.Path(root, 'resources', 'base.ui')
        if not file.exists():
            msg = f'Unable to locate UI file: {file.resolve().as_posix()}'
            raise FileNotFoundError(msg)
        return QFile(file.resolve().as_posix())

    def _init_mainwidget(self) -> None:
        '''Initializes a Qt main widget from the UI file.'''
        loader = QUiLoader()
        file = self._get_ui_file()
        
        file.open(QFile.ReadOnly)
        self.mainwidget = loader.load(file)
        file.close()
        self.init_ui()
        self.mainwidget.setWindowTitle('Pix2pix Trainer')

    def _set_stylesheet(self, app: QtWidgets.QApplication) -> None:
        root = pathlib.Path(__file__).parent
        file = pathlib.Path(root, 'resources', 'stylesheet.qss')
        if not file.exists():
            msg = f'Unable to locate stylesheet: {file.resolve().as_posix()}'
            raise FileNotFoundError(msg)
        with open(file.resolve().as_posix(), 'r') as file:
            stylesheet = file.read()
            app.setStyleSheet(stylesheet)

    def run(self) -> None:
        '''Entrypoint function for `Interface` class. Launches the GUI.'''
        app = QtWidgets.QApplication(sys.argv)
        self._set_stylesheet(app=app)

        self._init_mainwidget()
        self.mainwidget.show()

        sys.exit(app.exec())

