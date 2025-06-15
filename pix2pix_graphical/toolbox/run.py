import sys
import pathlib

from PySide6.QtWidgets import (
    QPushButton, QSlider, QComboBox, QApplication, QCheckBox, QFrame,
    QFileDialog, QSpinBox, QLineEdit)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QObject

from pix2pix_graphical.toolbox.helpers.config_helper import ConfigHelper
from pix2pix_graphical.toolbox.helpers.trainer_helper import TrainerHelper
from pix2pix_graphical.toolbox.helpers.tester_helper import TesterHelper
from pix2pix_graphical.toolbox.helpers.init_helper import InitHelper
from pix2pix_graphical.toolbox.components.log import OutputLog
from pix2pix_graphical.toolbox.components.settings_dock import SettingsDock
from pix2pix_graphical.toolbox.components.review_panel import ReviewPanel
from pix2pix_graphical.toolbox.components.utility_panel import UtilityPanel

import pix2pix_graphical.toolbox.utils.common as utils

class Interface(QObject):    
    def __init__(self) -> None:
        '''Init function for the `Interface` class.'''
        super().__init__()

        self.status_msg_length: int = 2000

    ### INIT INTERFACE ###

    def _init_ui_components(self) -> None:
        self.log = OutputLog(self.mainwidget, self.status_msg_length)
        self.confighelper = ConfigHelper(mainwidget=self.mainwidget)
        self.settings_dock = SettingsDock(mainwidget=self.mainwidget)
        self.testerhelper = TesterHelper(
            self.mainwidget, self.confighelper, self.log)
        self.trainerhelper = TrainerHelper(
            self.mainwidget, self.confighelper, 
            self.settings_dock, self.log, self.status_msg_length)
        self.reviewpanel = ReviewPanel(mainwidget=self.mainwidget)
        self.utilitypanel = UtilityPanel(mainwidget=self.mainwidget)
        
    def _load_from_config_file(self) -> None:
        selected = QFileDialog.getOpenFileName(filter='*.json')
        config_file = pathlib.Path(selected[0])
        if not config_file.suffix == '.json': return
        self.confighelper._init_from_config(config_path=config_file)

    def _testing_get_latest_epoch(self) -> None:
        exp_path = self.find(QLineEdit, 'test_experiment_path').text()
        latest_epoch = utils.get_latest_checkpoint_epoch(exp_path)
        self.find(QSpinBox, 'test_load_epoch').setValue(latest_epoch)

    def init_ui(self) -> None:
        '''Initialized the interface. Links parameters, sets defaults, etc.'''
        self._init_ui_components()
        self.settings_dock.init()
        self.inithelper.init(self.trainerhelper, self.settings_dock)

        self.find(QComboBox, 'direction').addItems(['AtoB', 'BtoA'])
        self.find( QPushButton, 'reload_stylesheet'
            ).clicked.connect(self._set_stylesheet)
        self.find(QComboBox, 'gen_block_type').addItems(
            ['UnetBlock', 'ResidualUnetBlock'])
        self.find(QComboBox, 'gen_upsample_type').addItems(
            ['Transposed', 'Bilinear'])
        crop_size = self.find(QComboBox, 'crop_size')
        crop_size.addItems(['16', '32', '64', '128', '256', '512', '1024'])
        crop_size.setCurrentIndex(4)
        self.find(QCheckBox, 'log_losses').clicked.connect(
            lambda x : self.find(QFrame, 'log_loss_settings').setEnabled(x))

        self.find(QComboBox, 'grayscale_method').addItems([
            'Weighted Average', 'From LAB', 'Desaturation', 
            'Average', 'Max', 'PCA'])
        self.find(QComboBox, 'compression_type').addItems(
            ['jpeg', 'webp'])
        self.find(QComboBox, 'optical_distortion_mode').addItems(
            ['Camera', 'Fisheye'])
        
        input_xforms = self.find(QFrame, 'input_transforms_frame')
        self.find(QCheckBox, 'show_input_transforms').clicked.connect(
            lambda x : input_xforms.setVisible(x))
        input_xforms.setVisible(False)

        both_xforms = self.find(QFrame, 'both_transforms_frame')
        self.find(QCheckBox, 'show_both_transforms').clicked.connect(
            lambda x : both_xforms.setVisible(x))
        both_xforms.setVisible(False)
        
        
        # REVIEW

        self.find(QPushButton, 'review_load_experiment').clicked.connect(
            self.reviewpanel.load_experiment)
        self.find(QSpinBox, 'review_graph_sample_rate').valueChanged.connect(
            lambda x : self.reviewpanel.set_graph_sample_rate(x))
        self.find(QComboBox, 'review_select_train_config').activated.connect(
            self.reviewpanel.load_train_config)

        # TESTING

        self.find(QPushButton, 'test_start'
            ).clicked.connect(self.testerhelper.start_test)
        self.find(QSlider, 'test_image_scale'
            ).valueChanged.connect(self.testerhelper.change_image_scale)
        self.find(QPushButton, 'test_image_scale_reset'
            ).clicked.connect(self.testerhelper.reset_image_scale)
        
        test_dataset_override = self.find(QFrame, 'test_dataset_path_frame')
        test_dataset_override.setHidden(True)
        self.find(QCheckBox, 'test_override_dataset').clicked.connect(
            lambda x : test_dataset_override.setHidden(not x))
        
        self.find(QPushButton, 'test_get_most_recent').clicked.connect(
            self._testing_get_latest_epoch)

        # Init from default config
        self.confighelper._init_from_config(config_path=None)
        self.find(QPushButton, 'load_from_config').clicked.connect(
            self._load_from_config_file)
        
        self.trainerhelper.safe_cleanup.connect(
            lambda : self.settings_dock.set_state('init'))

    ### INIT MAIN WIDGET ###

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

    ### ENTRYPOINT ###

    def run(self) -> None:
        '''Entrypoint function for `Interface` class. Launches the GUI.'''
        self.app = QApplication(sys.argv)
        self._set_stylesheet()

        self._init_mainwidget()
        self.find = self.mainwidget.findChild
        self.inithelper = InitHelper(mainwidget=self.mainwidget)
        self.inithelper._build_graphs()

        self.init_ui()
        self.mainwidget.show()

        sys.exit(self.app.exec())

