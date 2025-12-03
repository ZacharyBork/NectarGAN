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
class ConfigDataloaderStreaming:
    enable: bool
    dataset: str
    set: str
    subset: str
    max_samples: int
    cache_directory: int | None
    require_login: bool

@dataclass
class ConfigDataloader:
    dataroot: str
    batch_size: int
    num_workers: int
    streaming: ConfigDataloaderStreaming
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

##### DAE #####

@dataclass
class ConfigLearningRate:
    ramp_up: bool
    ramp_up_steps: int
    base_rate: float
    do_decay: bool
    steps_before_decay: int 
    decay_steps: int

@dataclass
class ConfigDAE:
    betas: list[float]
    time_embedding_dimension: int
    mlp_hidden_dimension: int
    mlp_output_dimension: int
    learning_rate: ConfigLearningRate
    in_channels: int
    features: int
    n_downs: int

##### NOISE_SCHEDULE #####

@dataclass
class ConfigNoiseSchedule:
    timesteps: int
    schedule_type: str
    cosine_offset: float

##### MODEL #####

@dataclass
class ConfigModel:
    model_type: str
    input_size: int
    mixed_precision: bool
    use_ema: bool
    ema_decay: float
    accumulate_gradients: bool
    gradient_accumulation_steps: int
    cfg_scale: float
    noise_schedule: ConfigNoiseSchedule
    dae: ConfigDAE

##### CAPTIONS #####

@dataclass
class ConfigCaptions:
    max_length: int
    use_fixed_captions: bool
    fixed_captions: list[str]
    metadata_file: str

##### LATENTS #####

@dataclass
class ConfigLatentCache:
    precache: bool
    runtime_cache: bool
    batch_size: int
    shard_size: int

@dataclass
class ConfigLatents:
    latent_size_divisor: int
    override_latent_size: bool
    latent_size: int
    caching: ConfigLatentCache

##### VISUALIZER #####

@dataclass
class ConfigVisdom:
    enable: bool
    average_loss: bool
    env_name: str
    server: str
    port: int
    image_size: int
    update_frequency: int

@dataclass
class ConfigConsole:
    average_loss: bool
    print_frequency: int

@dataclass
class ConfigVisualizer:
    visdom: ConfigVisdom
    console: ConfigConsole

##### MAIN #####

@dataclass
class DiffusionConfig(cfgcommon.Config):
    common: cfgcommon.ConfigCommon
    train: ConfigTrain
    model: ConfigModel
    dataloader: ConfigDataloader
    latents: ConfigLatents
    captions: ConfigCaptions
    save: cfgcommon.ConfigSave
    visualizer: ConfigVisualizer

    DEFAULT_FILE = 'defaults/diffusion.json'
    GROUP_SCHEMA = {
        'common': cfgcommon.ConfigCommon,
        'train': ConfigTrain,
        'model': ConfigModel,
        'dataloader': ConfigDataloader,
        'latents': ConfigLatents,
        'captions': ConfigCaptions,
        'save': cfgcommon.ConfigSave,
        'visualizer': ConfigVisualizer}
    
    def __post_init__(self) -> None:
        # This is hacky and needs to be fixed in the future.
        self.dataloader.load = cfgcommon.ConfigDataloaderLoad(
            load_size=self.model.input_size, crop_size=self.model.input_size, 
            input_nc=self.model.dae.in_channels)

