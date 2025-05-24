from dataclasses import dataclass

@dataclass
class ConfigCommon:
    device: str
    dataroot: str
    output_directory: str
    experiment_name: str
    direction: str
    input_nc: int

@dataclass
class ConfigDataloader:
    load_size: int
    crop_size: int
    batch_size: int
    num_workers: int
    both_flip_chance: float
    input_colorjitter_chance: float

@dataclass
class ConfigTrain:
    num_epochs: int
    num_epochs_decay: int
    learning_rate: float

@dataclass
class ConfigLoss:
    lambda_l1: float
    do_sobel_loss: bool
    lambda_sobel: float
    do_laplacian_loss: bool
    lambda_laplacian: float

@dataclass
class ConfigGenerator:
    upsample_mode: str

@dataclass
class ConfigDiscriminator:
    n_layers_d: int
    base_channels_d: int
    max_channels_d: int

@dataclass
class ConfigOptimizer:
    beta1: float

@dataclass
class ConfigLoad:
    continue_train: bool
    load_epoch: int

@dataclass
class ConfigSave:
    save_model: bool
    model_save_rate: int
    save_examples: bool
    example_save_rate: int
    num_examples: int

@dataclass
class ConfigVisualizer:
    update_frequency: int
    enable_visdom: bool
    visdom_env_name: str
    visdom_image_size: int

@dataclass
class Config:
    common: ConfigCommon
    dataloader: ConfigDataloader
    train: ConfigTrain
    loss: ConfigLoss
    generator: ConfigGenerator
    discriminator: ConfigDiscriminator
    optimizer: ConfigOptimizer
    load: ConfigLoad
    save: ConfigSave
    visualizer: ConfigVisualizer