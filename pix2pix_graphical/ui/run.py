import sys
import pathlib
from typing import Union

import PySide6.QtWidgets as QtWidgets
import PySide6.QtGui as QtGui
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import Qt, QFile, QObject

import pix2pix_graphical.ui.utils.common as utils

from pix2pix_graphical.ui.utils.config_helper import ConfigHelper
from pix2pix_graphical.ui.utils.trainer_helper import TrainerHelper
from pix2pix_graphical.ui.utils.log import OutputLog
from pix2pix_graphical.ui.utils.imagelabel import ImageLabel
from pix2pix_graphical.ui.utils.graphing import Graph
from pix2pix_graphical.ui.utils.settings_dock import SettingsDock


class Interface(QObject):    
    def __init__(self) -> None:
        '''Init function for the `Interface` class.'''
        super().__init__()

        # Stores references to frequently used widgets
        self.widgets: dict[str, QtWidgets.QWidget] = {}

        self.status_msg_length: int = 2000

    ### INIT HELPERS ###

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

    def _close_settings_dock(self) -> None:
        dock = self.find(QtWidgets.QDockWidget, 'settings_dock')
        button = self.find(QtWidgets.QPushButton, 'close_experiment_settings')
        direction = 'left' if dock.isHidden() else 'right'
        utils.set_button_icon(f'caret-line-{direction}-bold.svg', button)
        dock.setHidden(not dock.isHidden())

    def _split_lr_schedules(self) -> None:
        # gen_box = self.find(QtWidgets.QGroupBox, 'gen_schedule_box')
        disc_box = self.find(QtWidgets.QFrame, 'disc_schedule_box')
        checkbox = self.find(QtWidgets.QCheckBox, 'separate_lr_schedules')
        if checkbox.isChecked():
            # gen_box.setTitle('Generator')
            disc_box.setHidden(False)
        else:
            # gen_box.setTitle('Schedule')
            disc_box.setHidden(True)

    ### INIT INTERFACE ###

    def _get_signal_widgets(self) -> None:
        self.widgets = {
            'train_start': self.find(QtWidgets.QPushButton, 'train_start'),
            'train_stop': self.find(QtWidgets.QPushButton, 'train_stop'),
            'epoch_progress': self.find(QtWidgets.QProgressBar, 'epoch_progress'),
            'train_progress': self.find(QtWidgets.QProgressBar, 'train_progress'),
            'x_label': self.find(QtWidgets.QLabel, 'x_label'),
            'y_label': self.find(QtWidgets.QLabel, 'y_label'),
            'y_fake_label': self.find(QtWidgets.QLabel, 'y_fake_label')}

    def _init_training_controls(self) -> None:
        self.widgets['train_progress'].setValue(0)
        self.widgets['epoch_progress'].setValue(0)
        self.widgets['train_start'].clicked.connect(self.trainerhelper.start_train)
        self.widgets['train_stop'].clicked.connect(self.trainerhelper.stop_train)
        self.widgets['train_stop'].setEnabled(False)

        self.find(QtWidgets.QPushButton, 'train_pause').setHidden(True)
        self.find(QtWidgets.QPushButton, 'train_pause').clicked.connect(self.trainerhelper.pause_train)

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

        button = self.find(_button, 'close_experiment_settings')
        utils.set_button_icon('caret-line-left-bold.svg', button)
        button.clicked.connect(self._close_settings_dock)

    def _init_training_setting(self) -> None:
        self.find(QtWidgets.QFrame, 'disc_schedule_box').setHidden(True)
        self.find(QtWidgets.QCheckBox, 'separate_lr_schedules').clicked.connect(self._split_lr_schedules)

    def _init_update_frequency(self) -> None:
        update_freq = self.find(QtWidgets.QSpinBox, 'update_frequency')
        update_freq.valueChanged.connect(lambda x=update_freq.value() : self.trainerhelper._change_update_frequency(x))

    def _init_current_epoch_display(self) -> None:
        current_epoch = self.find(QtWidgets.QLCDNumber, 'current_epoch')
        current_epoch.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
        palette = current_epoch.palette()
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor('black'))
        current_epoch.setPalette(palette)
    
    def _build_graphs(self) -> None:
        performance = Graph(window_title='performance')
        performance.add_line(name='epoch_time', color=(255, 0, 0))
        performance.setObjectName('performance_graph')
        layout = self.mainwidget.findChild(QtWidgets.QVBoxLayout, 'performance_graph_layout')
        layout.addWidget(performance)


        loss_g = Graph(window_title='loss_g', bottom_label='Epoch')
        loss_g.add_line(name='G_GAN', color=(255, 0, 0))
        loss_g.add_line(name='G_L1', color=(0, 255, 0))
        loss_g.add_line(name='G_SOBEL', color=(0, 0, 255))
        loss_g.add_line(name='G_LAP', color=(255, 255, 0))
        loss_g.add_line(name='G_VGG', color=(255, 0, 255))
        loss_g.setObjectName('loss_g_graph')
        layout = self.mainwidget.findChild(QtWidgets.QVBoxLayout, 'loss_g_graph_layout')
        layout.addWidget(loss_g)

        loss_d = Graph(window_title='loss_d', bottom_label='Epoch')
        loss_d.add_line(name='D_real', color=(0, 255, 0))
        loss_d.add_line(name='D_fake', color=(255, 0, 0))
        loss_d.setObjectName('loss_d_graph')
        layout = self.mainwidget.findChild(QtWidgets.QVBoxLayout, 'loss_d_graph_layout')
        layout.addWidget(loss_d)
        
        self.graphs = {
            'performance': performance,
            'loss_g': loss_g,
            'loss_d': loss_d
        }

    def _set_context_dock_visibility(self) -> None:
        dock = self.find(QtWidgets.QDockWidget, 'context_settings_dock')
        open_dock = self.find(QtWidgets.QPushButton, 'open_context_settings')

        prefix = 'Show' if dock.isVisible() else 'Hide'
        open_dock.setText(f'{prefix} Visualizer Settings')
        dock.setVisible(not dock.isVisible())

    def _set_always_on_top(self, enabled: bool) -> None:
        self.mainwidget.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, enabled)
        self.mainwidget.show()

    def init_ui(self) -> None:
        '''Initialized the interface. Links parameters, sets defaults, etc.'''
        self._get_signal_widgets()     # Get commonly used widgets
        self._sliders_to_spinboxes()   # Link sliders to their spinboxes
        self._init_training_controls() # Link callbacks for training controls
        self._swap_image_labels()      # Replace QLabels with custom wrappers
        self._init_pushbuttons()
        self._init_training_setting()
        self._init_update_frequency()

        
        self.settings_dock.init()

        splitter = self.find(QtWidgets.QSplitter, 'visualizer_splitter')
        splitter.setSizes([1000 - 250, 250])


        self.find(QtWidgets.QPushButton, 'reload_stylesheet').clicked.connect(self._set_stylesheet)
        self.mainwidget.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.find(QtWidgets.QCheckBox, 'always_on_top').clicked.connect(lambda x : self._set_always_on_top(x))

        root = pathlib.Path(__file__).parent
        file = pathlib.Path(root, 'resources', 'icons', 'performance_graph_time_label.svg')
        transform = QtGui.QTransform()
        transform.rotate(270)
        self.find(QtWidgets.QLabel, 'perf_graph_time_label').setPixmap(
            QtGui.QPixmap(file.as_posix()).scaledToHeight(10).transformed(transform))
        
        file = pathlib.Path(root, 'resources', 'icons', 'performance_graph_epoch_label.svg')
        self.find(QtWidgets.QLabel, 'perf_graph_epoch_label').setPixmap(
            QtGui.QPixmap(file.as_posix()).scaledToHeight(10))

        self.find(
            QtWidgets.QPushButton, 'performance_graph_reset_framing'
        ).clicked.connect(lambda : self.graphs['performance'].reframe_graph())
        self.find(
            QtWidgets.QPushButton, 'performance_graph_clear'
        ).clicked.connect(lambda : self.graphs['performance'].reset_graph())

        self.find(QtWidgets.QComboBox, 'direction').addItems(['AtoB', 'BtoA'])
        
        self.confighelper._init_from_config(config_path=None) # Init from default config
        

    ### ENTRYPOINT ###

    def _get_ui_file(self) -> QFile:
        '''Gets the main widget's UI file and returns it as a QFile.
        
        Returns:
            QFile : The UI file.
        
        Raises:
            FileNotFoundError : If unable to locate UI file.
        '''
        root = pathlib.Path(__file__).parent
        file = pathlib.Path(root, 'resources', 'test.ui')
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
        self.mainwidget.setWindowTitle('NectarGAN Toolbox')

    def _set_stylesheet(self) -> None:
        root = pathlib.Path(__file__).parent
        file = pathlib.Path(root, 'resources', 'stylesheet.qss')
        if not file.exists():
            msg = f'Unable to locate stylesheet: {file.resolve().as_posix()}'
            raise FileNotFoundError(msg)
        with open(file.resolve().as_posix(), 'r') as file:
            stylesheet = file.read()
            self.app.setStyleSheet(stylesheet)

    def run(self) -> None:
        '''Entrypoint function for `Interface` class. Launches the GUI.'''
        self.app = QtWidgets.QApplication(sys.argv)
        self._set_stylesheet()

        self._init_mainwidget()
        self.find = self.mainwidget.findChild
        self._build_graphs()

        self.log = OutputLog(
            mainwidget=self.mainwidget,
            status_msg_length=self.status_msg_length)
        self.confighelper = ConfigHelper(mainwidget=self.mainwidget)
        self.trainerhelper = TrainerHelper(
            mainwidget=self.mainwidget,
            confighelper=self.confighelper,
            log=self.log, status_msg_length=self.status_msg_length)
        self.settings_dock = SettingsDock(mainwidget=self.mainwidget)

        self.init_ui()
        self.mainwidget.show()

        sys.exit(self.app.exec())

