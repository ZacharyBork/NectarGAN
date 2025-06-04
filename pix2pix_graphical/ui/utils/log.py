import pathlib
from dataclasses import dataclass

from PySide6.QtGui import QCursor
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QLabel, QMessageBox, QStatusBar, 
    QTextEdit, QPushButton, QCheckBox
)

@dataclass
class DefaultMessages:
    NO_EXPERIMENT_DIR: str=(
                f'No current experiement directory. '
                f'Please begin training before outputting log to file.')
    EXPORT_CONFIRM: str=(
                f'This will export the current log to a .txt file in the '
                f'experiment directory.\n\nPress Yes to continue.')
    CLEAR_LOG: str=(
                f'This will clear all data from the output log.\n'
                f'This action cannot be undone.\n\n'
                f'Are you sure you want to proceed?')

class OutputLog():
    def __init__(
            self, 
            mainwidget: QWidget,
            status_msg_length: int=2000
        ) -> None:
        self.mainwidget = mainwidget
        self.status_msg_length = status_msg_length
        
        self.dm = DefaultMessages()
        
        self.output_frozen: bool = False
        self.autoscroll_enabled: bool = True

        self._init_widgets()

    ### INIT HELPERS ###

    def _init_widgets(self) -> None:

        # Main Status bar
        self.statusbar = self.mainwidget.findChild(QStatusBar, 'statusbar')

        # Log
        self.log_widget = self.mainwidget.findChild(QTextEdit, 'output_log')
        self.log_widget.setReadOnly(True)
        self.log_widget.viewport().setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self.log_widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.log_widget.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)

        # Log Frozen Label
        self.frozen_label = self.mainwidget.findChild(QLabel, 'log_output_frozen_header')
        self.frozen_label.setHidden(True)
        
        # Log Function Buttons
        self.export_btn = self.mainwidget.findChild(QPushButton, 'export_output_log')
        self.export_btn.clicked.connect(self._export_output_log)
        self.freeze_btn = self.mainwidget.findChild(QPushButton, 'freeze_output_log')
        self.freeze_btn.clicked.connect(self._freeze_output_log)
        self.clear_btn = self.mainwidget.findChild(QPushButton, 'clear_output_log')
        self.clear_btn.clicked.connect(self._clear_output_log)

        # Log Option Checkboxes
        self.autoscroll_box = self.mainwidget.findChild(QCheckBox, 'autoscroll_log')     
        self.autoscroll_enabled = self.autoscroll_box.isChecked()

    ### READ / WRITE ###

    def write_entry(self, entry: str) -> None:
        self.log_widget.append(entry)

    ### LOG FUNCTION CALLBACKS ###

    def _freeze_output_log(self) -> None:
        if not self.log_frozen: self.log_widget.append('Output log frozen...')
        else: self.log_widget.append('Log unfrozen, resuming output...')
        self.frozen_label.setHidden(self.log_frozen)
        self.log_frozen = not self.log_frozen
        
    def _clear_output_log(self) -> None:
        if len(self.log_widget.toPlainText()) > 0:
            YES = QMessageBox.StandardButton.Yes
            NO = QMessageBox.StandardButton.No
            confirm = QMessageBox.question(
                self.mainwidget, 'Warning', self.dm.CLEAR_LOG, YES | NO)
            if confirm == YES: 
                self.log_widget.clear()
                message = 'Output Log Cleared...'
                self.statusbar.showMessage(message, self.status_msg_length)
            else: self.statusbar.showMessage('Canceled...', self.status_msg_length)
        else: 
            message = 'Output log empty...'
            self.statusbar.showMessage(message, self.status_msg_length)

    def _export_output_log(self, experiment_dir: str) -> None:
        if experiment_dir is None: 
            self.log_widget.append(self.dm.NO_EXPERIMENT_DIR)
            return
        
        log_string = self.log_widget.toPlainText()
        if len(log_string) > 0:
            YES = QMessageBox.StandardButton.Yes
            NO = QMessageBox.StandardButton.No
            
            confirm = QMessageBox.question(
                self.mainwidget, 'Please confirm', 
                self.dm.EXPORT_CONFIRM, YES | NO)
            
            if confirm == YES:
                if pathlib.Path(experiment_dir).exists(): 
                    output_dir = pathlib.Path(experiment_dir)
                else: self.log_widget.append(self.dm.NO_EXPERIMENT_DIR)
                
                log_file = pathlib.Path(output_dir, 'output_log_dump.txt')
                with open(log_file.as_posix(), 'w') as file:
                    file.write(log_string)
                
                message = f'Log Saved: {log_file.as_posix()}'
                self.statusbar.showMessage(message, self.status_msg_length)
            else: self.statusbar.showMessage('Canceled...', self.status_msg_length)
        else: 
            message = 'Output log empty, nothing to export...'
            self.statusbar.showMessage(message, self.status_msg_length)