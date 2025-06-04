from dataclasses import dataclass, field
from typing import Callable
from PySide6.QtWidgets import (
    QWidget, QLineEdit, QSpinBox, QDoubleSpinBox, 
    QComboBox, QCheckBox, QGroupBox)

@dataclass
class ConfigHelper:
    SETTERS: dict[
        type[QWidget], Callable
    ] = field(default_factory=dict)

    GETTERS: dict[
        type[QWidget], Callable
    ] = field(default_factory=dict)
    
    CONFIG_MAP: dict[
        str, tuple[str, type[QWidget]]
    ] = field(default_factory=dict)

    def __post_init__(self):
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
            'continue_train': ('train.load.continue_train', QGroupBox),
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
            'visdom_enable': ('visualizer.visdom.enable', QGroupBox),
            'visdom_env_name': ('visualizer.visdom.env_name', QLineEdit),
            'visdom_port': ('visualizer.visdom.port', QSpinBox),
            'visdom_image_size': ('visualizer.visdom.image_size', QSpinBox),
            'visdom_update_frequency': ('visualizer.visdom.update_frequency', QSpinBox)}
        