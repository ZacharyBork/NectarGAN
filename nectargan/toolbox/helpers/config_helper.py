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
            path = files('nectargan.config').joinpath('default.json')
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
            config_keys, id = info
            widget_type = self.SETTERS[id][0]
            keys = config_keys.split('.')
            value = self.config
            for key in keys: value = value[key]
            
            widget = self.mainwidget.findChild(widget_type, ui_name)
            self.SETTERS[id][1](widget, value)

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
            config_keys, id = info
            widget_type = self.GETTERS[id][0]
            keys = config_keys.split('.')
            config_entry = self.config
            for key in keys[:-1]: config_entry = config_entry[key]
            
            widget = self.mainwidget.findChild(widget_type, ui_name)
            config_entry[keys[-1]] = self.GETTERS[id][1](widget)

    def _build_mappings(self) -> None:
        self.SETTERS = {
            'QLineEdit': (
                QLineEdit,
                lambda widget, value : widget.setText(value)),
            'QSpinBox': (
                QSpinBox,
                lambda widget, value : widget.setValue(value)),
            'QDoubleSpinBox': (
                QDoubleSpinBox,
                lambda widget, value : widget.setValue(value)),
            'QDoubleSpinBox_percent': (
                QDoubleSpinBox,
                lambda widget, value : widget.setValue(100.0 * value)),
            'QComboBox': (
                QComboBox,
                lambda widget, value : widget.setCurrentText(value)),
            'QCheckBox': (
                QCheckBox,
                lambda widget, value : widget.setChecked(value)),
            'QGroupBox': (
                QGroupBox,
                lambda widget, value : widget.setChecked(value)),
            'crop_size': (
                QComboBox,
                lambda widget, value : widget.setCurrentText(str(value))),
            'grayscale_method':(
                QComboBox,
                lambda widget, value : widget.setCurrentText(
                    value.replace('_', ' ').capitalize()))}
        
        self.GETTERS = { 
            'QLineEdit': (
                QLineEdit,
                lambda widget : widget.text()),
            'QSpinBox': (
                QSpinBox,
                lambda widget : widget.value()),
            'QDoubleSpinBox': (
                QDoubleSpinBox,
                lambda widget : widget.value()),
            'QDoubleSpinBox_percent': (
                QDoubleSpinBox,
                lambda widget : 0.01 * widget.value()),
            'QComboBox': (
                QComboBox,
                lambda widget : widget.currentText()),
            'QCheckBox': (
                QCheckBox,
                lambda widget : widget.isChecked()),
            'QGroupBox': (
                QGroupBox,
                lambda widget : widget.isChecked()),
            'crop_size': (
                QComboBox,
                lambda widget : int(widget.currentText())),
            'grayscale_method':(
                QComboBox,
                lambda widget : 
                    widget.currentText().lower().replace(' ', '_'))}

        self.CONFIG_MAP = {
            # Experiment
            'output_directory': (
                'common.output_directory', 'QLineEdit'),
            'experiment_name': (
                'common.experiment_name', 'QLineEdit'),
            'experiment_version': (
                'common.experiment_version', 'QSpinBox'),

            # Generator Architecture
            'gen_features': (
                'train.generator.features', 'QSpinBox'),
            'gen_n_downs': (
                'train.generator.n_downs', 'QSpinBox'),
            'gen_block_type': (
                'train.generator.block_type', 'QComboBox'),
            'gen_upsample_type': (
                'train.generator.upsample_type', 'QComboBox'),

            # Discriminator Architecture
            'n_layers_d': (
                'train.discriminator.n_layers', 'QSpinBox'), 
            'base_channels_d': (
                'train.discriminator.base_channels', 'QSpinBox'),
            'max_channels_d': (
                'train.discriminator.max_channels', 'QSpinBox'),

            # Dataloader
            'dataroot': ('dataloader.dataroot', 'QLineEdit'),
            'direction': ('dataloader.direction', 'QComboBox'),
            'batch_size': ('dataloader.batch_size', 'QSpinBox'),
            'input_nc': ('dataloader.load.input_nc', 'QSpinBox'),
            'load_size': ('dataloader.load.load_size', 'QSpinBox'),
            'crop_size': ('dataloader.load.crop_size', 'crop_size'),

            # Augmentations
            'colorjitter_chance': (
                'dataloader.augmentations.input.colorjitter_chance', 
                'QDoubleSpinBox_percent'),
            'colorjitter_min_brightness': (
                'dataloader.augmentations.input.colorjitter_min_brightness', 
                'QDoubleSpinBox'),
            'colorjitter_max_brightness': (
                'dataloader.augmentations.input.colorjitter_max_brightness', 
                'QDoubleSpinBox'),
            
            'gaussnoise_chance': (
                'dataloader.augmentations.input.gaussnoise_chance', 
                'QDoubleSpinBox_percent'),
            'gaussnoise_min': (
                'dataloader.augmentations.input.gaussnoise_min', 
                'QDoubleSpinBox'),
            'gaussnoise_max': (
                'dataloader.augmentations.input.gaussnoise_max', 
                'QDoubleSpinBox'),

            'motionblur_chance': (
                'dataloader.augmentations.input.motionblur_chance', 
                'QDoubleSpinBox_percent'),
            'motionblur_limit': (
                'dataloader.augmentations.input.motionblur_limit', 
                'QSpinBox'),

            'randgamma_chance': (
                'dataloader.augmentations.input.randgamma_chance', 
                'QDoubleSpinBox_percent'),
            'randgamma_min': (
                'dataloader.augmentations.input.randgamma_min', 
                'QDoubleSpinBox'),
            'randgamma_max': (
                'dataloader.augmentations.input.randgamma_max', 
                'QDoubleSpinBox'),

            'grayscale_chance': (
                'dataloader.augmentations.input.grayscale_chance', 
                'QDoubleSpinBox_percent'),
            'grayscale_method': (
                'dataloader.augmentations.input.grayscale_method', 
                'grayscale_method'),

            'compression_chance': (
                'dataloader.augmentations.input.compression_chance', 
                'QDoubleSpinBox_percent'),
            'compression_type': (
                'dataloader.augmentations.input.compression_type', 
                'QComboBox'),
            'compression_quality_min': (
                'dataloader.augmentations.input.compression_quality_min', 
                'QSpinBox'),
            'compression_quality_max': (
                'dataloader.augmentations.input.compression_quality_max', 
                'QSpinBox'),

            'h_flip_chance': (
                'dataloader.augmentations.both.h_flip_chance', 
                'QDoubleSpinBox_percent'),
            'v_flip_chance': (
                'dataloader.augmentations.both.v_flip_chance', 
                'QDoubleSpinBox_percent'),
            'rot90_chance': (
                'dataloader.augmentations.both.rot90_chance', 
                'QDoubleSpinBox_percent'),

            'elastic_transform_chance': (
                'dataloader.augmentations.both.elastic_transform_chance', 
                'QDoubleSpinBox_percent'),
            'elastic_transform_alpha': (
                'dataloader.augmentations.both.elastic_transform_alpha', 
                'QDoubleSpinBox'),
            'elastic_transform_sigma': (
                'dataloader.augmentations.both.elastic_transform_sigma', 
                'QDoubleSpinBox'),

            'optical_distortion_chance': (
                'dataloader.augmentations.both.optical_distortion_chance', 
                'QDoubleSpinBox_percent'),
            'optical_distortion_min': (
                'dataloader.augmentations.both.optical_distortion_min', 
                'QDoubleSpinBox'),
            'optical_distortion_max': (
                'dataloader.augmentations.both.optical_distortion_max', 
                'QDoubleSpinBox'),
            'optical_distortion_mode': (
                'dataloader.augmentations.both.optical_distortion_mode', 
                'grayscale_method'),
            
            'coarse_dropout_chance': (
                'dataloader.augmentations.both.optical_distortion_chance', 
                'QDoubleSpinBox_percent'),

            'coarse_dropout_holes_min': (
                'dataloader.augmentations.both.coarse_dropout_holes_min', 
                'QSpinBox'),
            'coarse_dropout_holes_max': (
                'dataloader.augmentations.both.coarse_dropout_holes_max', 
                'QSpinBox'),

            'coarse_dropout_height_min': (
                'dataloader.augmentations.both.coarse_dropout_height_min', 
                'QDoubleSpinBox'),
            'coarse_dropout_height_max': (
                'dataloader.augmentations.both.coarse_dropout_height_max', 
                'QDoubleSpinBox'),

            'coarse_dropout_width_min': (
                'dataloader.augmentations.both.coarse_dropout_width_min', 
                'QDoubleSpinBox'),
            'coarse_dropout_width_max': (
                'dataloader.augmentations.both.coarse_dropout_width_max', 
                'QDoubleSpinBox'),

            # Train 
            'separate_lr_schedules': (
                'train.separate_lr_schedules', 'QCheckBox'),
            'continue_train': (
                'train.load.continue_train', 'QCheckBox'),
            'load_epoch': (
                'train.load.load_epoch', 'QSpinBox'),

            # Generator
            'gen_epochs': (
                'train.generator.learning_rate.epochs', 'QSpinBox'),
            'gen_epochs_decay': (
                'train.generator.learning_rate.epochs_decay', 'QSpinBox'),
            'gen_lr_initial': (
                'train.generator.learning_rate.initial', 'QDoubleSpinBox'),
            'gen_lr_target': (
                'train.generator.learning_rate.target', 'QDoubleSpinBox'),
            'gen_optim_beta1': (
                'train.generator.optimizer.beta1', 'QDoubleSpinBox'),

            # Discriminator
            'disc_epochs': (
                'train.discriminator.learning_rate.epochs', 'QSpinBox'),
            'disc_epochs_decay': (
                'train.discriminator.learning_rate.epochs_decay', 'QSpinBox'),
            'disc_lr_initial': (
                'train.discriminator.learning_rate.initial', 'QDoubleSpinBox'),
            'disc_lr_target': (
                'train.discriminator.learning_rate.target', 'QDoubleSpinBox'),
            'disc_optim_beta1': (
                'train.discriminator.optimizer.beta1', 'QDoubleSpinBox'),

            # Loss
            'lambda_gan': (
                'train.loss.lambda_gan', 'QDoubleSpinBox'),
            'lambda_l1': (
                'train.loss.lambda_l1', 'QDoubleSpinBox'),
            'lambda_l2': (
                'train.loss.lambda_l2', 'QDoubleSpinBox'),
            'lambda_sobel': (
                'train.loss.lambda_sobel', 'QDoubleSpinBox'),
            'lambda_laplacian': (
                'train.loss.lambda_laplacian', 'QDoubleSpinBox'),
            'lambda_vgg': (
                'train.loss.lambda_vgg', 'QDoubleSpinBox'),

            # Save
            'save_model': (
                'save.save_model', 'QCheckBox'),
            'model_save_rate': (
                'save.model_save_rate', 'QSpinBox'),
            'save_examples': (
                'save.save_examples', 'QCheckBox'),
            'example_save_rate': (
                'save.example_save_rate', 'QSpinBox'),
            'num_examples': (
                'save.num_examples', 'QSpinBox')}