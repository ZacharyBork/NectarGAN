from typing import Union

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QDockWidget, QPushButton, QFrame, QCheckBox, QSlider, QSpinBox, 
    QDoubleSpinBox
)

import pix2pix_graphical.toolbox.utils.common as utils
from pix2pix_graphical.toolbox.components.settings_dock import SettingsDock
from pix2pix_graphical.toolbox.helpers.trainer_helper import TrainerHelper

class Callbacks():
    def __init__(self, mainwidget: QWidget):
        self.mainwidget = mainwidget
        self.find = self.mainwidget.findChild

    def _start_train(
            self, 
            trainerhelper: TrainerHelper, 
            settings_dock: SettingsDock
        ) -> None:
        '''Starts training and sets settings dock lock state.'''
        trainerhelper.start_train()
        settings_dock.set_state('training')

    def _stop_train(
            self, 
            trainerhelper: TrainerHelper, 
            settings_dock: SettingsDock
    ) -> None:
        '''Stops training and resets the settings dock state.'''
        trainerhelper.stop_train()
        settings_dock.set_state('init')

    def _close_settings_dock(self) -> None:
        '''Toggles visibility of the main settings dock.'''
        dock = self.find(QDockWidget, 'settings_dock')
        button = self.find(QPushButton, 'close_experiment_settings')
        direction = 'left' if dock.isHidden() else 'right'
        utils.set_button_icon(f'caret-line-{direction}-bold.svg', button)
        dock.setHidden(not dock.isHidden())

    def _split_lr_schedules(self) -> None:
        '''Shows or hides Discriminator schedule settings.'''
        disc_box = self.find(QFrame, 'disc_schedule_box')
        checkbox = self.find(QCheckBox, 'separate_lr_schedules')
        disc_box.setHidden(not checkbox.isChecked())

    def _set_always_on_top(self, enabled: bool) -> None:
        '''Toggles always-on-top state on main window.
        
        Args:
            enabled : If true, window will stay on top. Otherwise it will not.
        '''
        self.mainwidget.setWindowFlag(
            Qt.WindowType.WindowStaysOnTopHint, enabled)
        self.mainwidget.show()

    def _update_slider_from_spinbox(
            self,
            slider: QSlider, 
            spinbox: Union[QSpinBox, QDoubleSpinBox], 
            multiplier: float=1.0
        ) -> None:
        '''Sets the value of a QSlider to the value of a QSpinbox-type widget.
        '''
        value = float(spinbox.value()) * multiplier
        value = max(spinbox.minimum(), min(spinbox.value(), spinbox.maximum))
        slider.setValue(value)

    def _update_spinbox_from_slider(
            self,
            spinbox: Union[QSpinBox, QDoubleSpinBox], 
            slider: QSlider, 
            multiplier: float=1.0
        ) -> None:
        '''Sets the value of a QSpinbox-type widget to the value of a QSlider.
        '''
        spinbox.setValue(float(slider.value()) * multiplier)
