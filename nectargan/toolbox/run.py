import sys
import pathlib

from PySide6.QtWidgets import (
    QPushButton, QApplication, QFileDialog)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QObject
from PySide6.QtGui import QShortcut, QKeySequence

from nectargan.toolbox.helpers.config_helper import ConfigHelper
from nectargan.toolbox.helpers.trainer_helper import TrainerHelper
from nectargan.toolbox.helpers.tester_helper import TesterHelper
from nectargan.toolbox.helpers.init_helper import InitHelper
from nectargan.toolbox.components.log import OutputLog
from nectargan.toolbox.components.settings_dock import SettingsDock
from nectargan.toolbox.components.review_panel import ReviewPanel
from nectargan.toolbox.components.utility_panel import UtilityPanel

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

    def _map_hotkeys(self) -> None:
        QShortcut(QKeySequence('Ctrl+Shift+R'), self.mainwidget
            ).activated.connect(self._set_stylesheet)
        QShortcut(QKeySequence('Ctrl+L'), self.mainwidget
            ).activated.connect(self._load_from_config_file)
        names = [
            'experiment', 'dataset', 'training', 
            'testing', 'review', 'utils', 'settings']
        for i in range(7):
            button = self.find(QPushButton, f'{names[i]}_settings_btn')
            QShortcut(QKeySequence(f'Ctrl+{i+1}'), self.mainwidget
                ).activated.connect(button.click)

    def init_ui(self) -> None:
        '''Initialized the interface. Links parameters, sets defaults, etc.'''
        self._init_ui_components()
        self.settings_dock.init()
        self.inithelper.init(self.trainerhelper, self.settings_dock)
        self._map_hotkeys()

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

