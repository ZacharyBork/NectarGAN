from dataclasses import dataclass
import nectargan.config.data.common as cfgcommon

##### DATALOADER #####

@dataclass
class ConfigAugsBoth:
    h_flip_chance: float
    v_flip_chance: float
    rot90_chance: float
    elastic_transform_chance: float
    elastic_transform_alpha: float
    elastic_transform_sigma: float
    optical_distortion_chance: float
    optical_distortion_min: float
    optical_distortion_max: float
    optical_distortion_mode: str
    coarse_dropout_chance: float
    coarse_dropout_holes_min: int
    coarse_dropout_holes_max: int
    coarse_dropout_height_min: float
    coarse_dropout_height_max: float
    coarse_dropout_width_min: float
    coarse_dropout_width_max: float

@dataclass
class ConfigAugsInput:
    colorjitter_chance: float
    colorjitter_min_brightness: float
    colorjitter_max_brightness: float
    gaussnoise_chance: float
    gaussnoise_min: float
    gaussnoise_max: float
    motionblur_chance: float
    motionblur_limit: int
    randgamma_chance: float
    randgamma_min: float
    randgamma_max: float
    grayscale_chance: float
    grayscale_method:  str
    compression_chance: float
    compression_type: str
    compression_quality_min: int
    compression_quality_max: int

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
    dataroot: str
    direction: str
    batch_size: int
    num_workers: int
    load: cfgcommon.ConfigDataloaderLoad
    augmentations: ConfigDataLoaderAugmentations

##### TRAIN #####

@dataclass
class ConfigOptimizer:
    beta1: float

@dataclass
class ConfigGenerator:
    features: int
    n_downs: int
    block_type: str
    upsample_type: str
    learning_rate: cfgcommon.ConfigLearningRate
    optimizer: ConfigOptimizer
    
@dataclass
class ConfigDiscriminator:
    n_layers: int
    base_channels: int
    max_channels: int
    learning_rate: cfgcommon.ConfigLearningRate
    optimizer: ConfigOptimizer

@dataclass
class ConfigLoss:
    lambda_gan: float
    lambda_l1: float
    lambda_l2: float
    lambda_sobel: float
    lambda_laplacian: float
    lambda_vgg: float

@dataclass
class ConfigTrain:
    separate_lr_schedules: bool
    load: cfgcommon.ConfigLoad
    generator: ConfigGenerator
    discriminator: ConfigDiscriminator
    loss: ConfigLoss

##### VISUALIZER #####

@dataclass
class ConfigVisualizer:
    visdom: cfgcommon.ConfigVisdom

##### MAIN #####

@dataclass
class GANConfig(cfgcommon.Config):
    common: cfgcommon.ConfigCommon
    dataloader: ConfigDataloader
    train: ConfigTrain
    save: cfgcommon.ConfigSave
    visualizer: ConfigVisualizer

    DEFAULT_FILE = 'default.json'
    GROUP_SCHEMA = {
        'common': cfgcommon.ConfigCommon,
        'dataloader': ConfigDataloader,
        'train': ConfigTrain,
        'save': cfgcommon.ConfigSave,
        'visualizer': ConfigVisualizer}
