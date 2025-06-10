import json
import pathlib
from typing import Callable, Any
from importlib.resources import files

from PySide6.QtWidgets import (
    QWidget, QLineEdit, QSpinBox, QDoubleSpinBox, 
    QComboBox, QCheckBox, QGroupBox)

class ConfigHelper:
    def __init__(self, mainwidget: QWidget) -> None:
        self.mainwidget = mainwidget
        self.config: dict[str, Any] | None = None

        SETTERS: dict[type[QWidget], Callable] = {}
        GETTERS: dict[type[QWidget], Callable] = {}
        CONFIG_MAP: dict[str, tuple[str, type[QWidget]]] = {}
        self._build_mappings() # Init SETTERS, GETTERS, and CONFIG_MAP

    def _get_config_data(
            self, 
            path: pathlib.Path | None=None
        ) -> dict[str, Any]:
        if path is None:
            path = files('pix2pix_graphical.config').joinpath('default.json')
        else: path = path.as_posix()
        try:
            with open(path, 'r') as f:
                data = json.load(f)['config']
            return data
        except Exception as e:
            message = f'Unable to open config file at: {path.as_posix()}'
            raise RuntimeError(message) from e

    def _init_from_config(self, config_path: pathlib.Path | None=None) -> None:
        self.config = self._get_config_data(config_path)
        for ui_name, info in self.CONFIG_MAP.items():
            config_keys, widget_type = info
            keys = config_keys.split('.')
            value = self.config
            for key in keys: value = value[key]
            
            widget = self.mainwidget.findChild(widget_type, ui_name)
            self.SETTERS[widget_type](widget, value)

    def _overwrite_disc_lr(self) -> None:
        names = [
            ('_epochs', QSpinBox), 
            ('_epochs_decay', QSpinBox), 
            ('_lr_initial', QDoubleSpinBox), 
            ('_lr_target', QDoubleSpinBox)]        
        for x, type in names:
            gen, disc = 'gen', 'disc'
            value = self.mainwidget.findChild(type, f'{gen}{x}').value()
            self.mainwidget.findChild(type, f'{disc}{x}').setValue(value)

    def _build_launch_config(self):
        split = self.mainwidget.findChild(
            QCheckBox, 'separate_lr_schedules').isChecked()
        if not split: self._overwrite_disc_lr()
        for ui_name, info in self.CONFIG_MAP.items():
            config_keys, widget_type = info
            keys = config_keys.split('.')
            config_entry = self.config
            for key in keys[:-1]: config_entry = config_entry[key]
            
            widget = self.mainwidget.findChild(widget_type, ui_name)
            config_entry[keys[-1]] = self.GETTERS[widget_type](widget)


    def _build_mappings(self) -> None:
        self.SETTERS = {
            QLineEdit: lambda widget, value : widget.setText(value),
            QSpinBox: lambda widget, value : widget.setValue(value),
            QDoubleSpinBox: lambda widget, value : widget.setValue(value),
            QComboBox: lambda widget, value : widget.setCurrentText(value),
            QCheckBox: lambda widget, value : widget.setChecked(value),
            QGroupBox: lambda widget, value : widget.setChecked(value)}
        
        self.GETTERS = { 
            QLineEdit: lambda widget : widget.text(),
            QSpinBox: lambda widget : widget.value(),
            QDoubleSpinBox: lambda widget : widget.value(),
            QComboBox: lambda widget : widget.currentText(),
            QCheckBox: lambda widget : widget.isChecked(),
            QGroupBox: lambda widget : widget.isChecked()}

        self.CONFIG_MAP = {
            # Experiment
            'output_directory': ('common.output_directory', QLineEdit),
            'experiment_name': ('common.experiment_name', QLineEdit),
            'experiment_version': ('common.experiment_version', QSpinBox),

            # Dataloader
            'dataroot': ('dataloader.dataroot', QLineEdit),
            'direction': ('dataloader.direction', QComboBox),
            'batch_size': ('dataloader.batch_size', QSpinBox),
            'input_nc': ('dataloader.load.input_nc', QSpinBox),
            'load_size': ('dataloader.load.load_size', QSpinBox),
            'crop_size': ('dataloader.load.crop_size', QSpinBox),

            # Train 
            'separate_lr_schedules': ('train.separate_lr_schedules', QCheckBox),
            'continue_train': ('train.load.continue_train', QCheckBox),
            'load_epoch': ('train.load.load_epoch', QSpinBox),

            # Generator
            'gen_epochs': ('train.generator.learning_rate.epochs', QSpinBox),
            'gen_epochs_decay': ('train.generator.learning_rate.epochs_decay', QSpinBox),
            'gen_lr_initial': ('train.generator.learning_rate.initial', QDoubleSpinBox),
            'gen_lr_target': ('train.generator.learning_rate.target', QDoubleSpinBox),
            'gen_optim_beta1': ('train.generator.optimizer.beta1', QDoubleSpinBox),

            # Discriminator
            'disc_epochs': ('train.discriminator.learning_rate.epochs', QSpinBox),
            'disc_epochs_decay': ('train.discriminator.learning_rate.epochs_decay', QSpinBox),
            'disc_lr_initial': ('train.discriminator.learning_rate.initial', QDoubleSpinBox),
            'disc_lr_target': ('train.discriminator.learning_rate.target', QDoubleSpinBox),
            'disc_optim_beta1': ('train.discriminator.optimizer.beta1', QDoubleSpinBox),

            # Loss
            'lambda_l1': ('train.loss.lambda_l1', QDoubleSpinBox),
            'lambda_sobel': ('train.loss.lambda_sobel', QDoubleSpinBox),
            'lambda_laplacian': ('train.loss.lambda_laplacian', QDoubleSpinBox),
            'lambda_vgg': ('train.loss.lambda_vgg', QDoubleSpinBox),

            # Save
            'save_model': ('save.save_model', QCheckBox),
            'model_save_rate': ('save.model_save_rate', QSpinBox),
            'save_examples': ('save.save_examples', QCheckBox),
            'example_save_rate': ('save.example_save_rate', QSpinBox),
            'num_examples': ('save.num_examples', QSpinBox),

            # Visdom
            'visdom_enable': ('visualizer.visdom.enable', QCheckBox),
            'visdom_env_name': ('visualizer.visdom.env_name', QLineEdit),
            'visdom_port': ('visualizer.visdom.port', QSpinBox),
            'visdom_image_size': ('visualizer.visdom.image_size', QSpinBox),
            'visdom_update_frequency': ('visualizer.visdom.update_frequency', QSpinBox)}
        