from dataclasses import dataclass

@dataclass
class ConfigCommon:
    device: str
    gpu_ids: list[int]
    output_directory: str
    experiment_name: str
    experiment_version: int

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
class ConfigDataloaderLoad:
    load_size: int
    crop_size: int
    input_nc: int
    
@dataclass
class ConfigDataloader:
    dataroot: str
    direction: str
    batch_size: int
    num_workers: int
    load: ConfigDataloaderLoad
    augmentations: ConfigDataLoaderAugmentations

@dataclass
class ConfigLoad:
    continue_train: bool
    load_epoch: int

@dataclass
class ConfigLearningRate:
    epochs: int
    epochs_decay: int
    initial: float
    target: float

@dataclass
class ConfigOptimizer:
    beta1: float

@dataclass
class ConfigGenerator:
    features: int
    n_downs: int
    block_type: str
    upsample_type: str
    learning_rate: ConfigLearningRate
    optimizer: ConfigOptimizer
    
@dataclass
class ConfigDiscriminator:
    n_layers: int
    base_channels: int
    max_channels: int
    learning_rate: ConfigLearningRate
    optimizer: ConfigOptimizer

@dataclass
class ConfigLoss:
    lambda_gan: float
    lambda_l1: float
    lambda_sobel: float
    lambda_laplacian: float
    lambda_vgg: float

@dataclass
class ConfigTrain:
    separate_lr_schedules: bool
    load: ConfigLoad
    generator: ConfigGenerator
    discriminator: ConfigDiscriminator
    loss: ConfigLoss

@dataclass
class ConfigSave:
    save_model: bool
    model_save_rate: int
    auto_increment_version: bool
    save_examples: bool
    example_save_rate: int
    num_examples: int

@dataclass
class ConfigVisdom:
    enable: bool
    env_name: str
    port: int
    image_size: int
    update_frequency: int
    
@dataclass
class ConfigVisualizer:
    visdom: ConfigVisdom

@dataclass
class Config:
    common: ConfigCommon
    dataloader: ConfigDataloader
    train: ConfigTrain
    save: ConfigSave
    visualizer: ConfigVisualizer