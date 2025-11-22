from dataclasses import dataclass
import nectargan.config.data.common as cfgcommon

##### DATALOADER #####

@dataclass
class ConfigAugmentations:
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
class ConfigDataloader:
    dataroot: str
    batch_size: int
    num_workers: int
    load: cfgcommon.ConfigDataloaderLoad
    augmentations: ConfigAugmentations

##### TRAIN #####

@dataclass
class ConfigLoss:
    lambda_mse: float

@dataclass
class ConfigTrain:
    load: cfgcommon.ConfigLoad
    loss: ConfigLoss

##### MODEL #####

@dataclass
class ConfigDAECommon:
    betas: list[float]
    time_embedding_dimension: int
    mlp_hidden_dimension: int
    learning_rate: cfgcommon.ConfigLearningRate

@dataclass
class ConfigDAEPixel:
    in_channels: int
    features: int
    n_downs: int

@dataclass
class ConfigDAELatent:
    in_channels: int
    features: int
    n_downs: int

@dataclass
class ConfigLatentPrecache:
    enable: bool
    batch_size: int
    shard_size: int

@dataclass
class ConfigModelCommon:
    timesteps: int
    noise_schedule: str
    cosine_offset: float
    dae: ConfigDAECommon

@dataclass
class ConfigModelPixel:
    input_size: int
    dae: ConfigDAEPixel

@dataclass
class ConfigModelLatent:
    input_size: int
    latent_size_divisor: int
    override_latent_size: bool
    latent_size: int
    precache: ConfigLatentPrecache
    dae: ConfigDAELatent

@dataclass
class ConfigModelStable:
    input_size: int
    latent_size_divisor: int
    override_latent_size: bool
    latent_size: int
    precache: ConfigLatentPrecache
    dae: ConfigDAELatent

@dataclass
class ConfigModel:
    model_type: str
    mixed_precision: bool
    use_ema: bool
    common: ConfigModelCommon
    pixel: ConfigModelPixel
    latent: ConfigModelLatent
    stable: ConfigModelStable

##### VISUALIZER #####

@dataclass
class ConfigConsole:
    print_frequency: int

@dataclass
class ConfigVisualizer:
    visdom: cfgcommon.ConfigVisdom
    console: ConfigConsole

##### MAIN #####

@dataclass
class DiffusionConfig(cfgcommon.Config):
    common: cfgcommon.ConfigCommon
    dataloader: ConfigDataloader
    train: ConfigTrain
    model: ConfigModel
    save: cfgcommon.ConfigSave
    visualizer: ConfigVisualizer

    DEFAULT_FILE = 'defaults/diffusion.json'
    GROUP_SCHEMA = {
        'common': cfgcommon.ConfigCommon,
        'dataloader': ConfigDataloader,
        'train': ConfigTrain,
        'model': ConfigModel,
        'save': cfgcommon.ConfigSave,
        'visualizer': ConfigVisualizer}
    
    def __post_init__(self) -> None:
        # This is hacky and needs to be fixed in the future.
        match self.model.model_type:
            case 'pixel': m = self.model.pixel
            case 'latent': m = self.model.latent
            case 'stable': m = self.model.stable
            case _: raise ValueError(
                f'Invalid model_type: {self.model.model_type}')
        self.dataloader.load = cfgcommon.ConfigDataloaderLoad(
            load_size=m.input_size, crop_size=m.input_size, 
            input_nc=m.dae.in_channels)

