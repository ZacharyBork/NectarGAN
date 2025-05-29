from dataclasses import dataclass

@dataclass
class ConfigCommon:
    device: str
    dataroot: str
    output_directory: str
    experiment_name: str
    experiment_version: int
    direction: str
    input_nc: int

@dataclass
class ConfigAugsBoth:
    h_flip_chance: float
    v_flip_chance: float
    rot90_chance: float

@dataclass
class ConfigAugsInput:
    colorjitter_chance: float
    colorjitter_brightness: list[float]

@dataclass
class ConfigAugsOutput:
    dummy: float

@dataclass
class ConfigDataLoaderAugmentations:
    both: ConfigAugsBoth
    input: ConfigAugsInput
    output: ConfigAugsOutput

@dataclass
class ConfigDataloader:
    load_size: int
    crop_size: int
    batch_size: int
    num_workers: int
    augmentations: ConfigDataLoaderAugmentations

@dataclass
class ConfigTrain:
    num_epochs: int
    num_epochs_decay: int
    learning_rate: float

@dataclass
class ConfigLoss:
    lambda_gan: float
    lambda_l1: float
    lambda_sobel: float
    lambda_laplacian: float
    lambda_vgg: float

@dataclass
class ConfigGenerator:
    upsample_block_type: str

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
    auto_increment_version: bool
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