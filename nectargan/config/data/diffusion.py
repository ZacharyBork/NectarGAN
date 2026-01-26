from dataclasses import dataclass
import nectargan.config.data.common as cfgcommon

##### DATALOADER #####

@dataclass
class ConfigAugmentations:
    h_flip_chance: float
    v_flip_chance: float
    rot90_chance: float
    
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
class ConfigDataloaderLoad:
    load_size: int
    crop_size: int
    input_nc: int

@dataclass
class ConfigDataloader:
    dataroot: str
    batch_size: int
    num_workers: int
    crop_type: str | None
    drop_last: bool
    pin_memory: bool
    shuffle: bool
    streaming: ConfigDataloaderStreaming
    load: ConfigDataloaderLoad
    augmentations: ConfigAugmentations

##### TRAIN #####

@dataclass
class ConfigLoad:
    continue_train: bool
    load_step: int

@dataclass
class ConfigPixelLoss:
    frequency: int
    max_batches: int
    lambda_l1: float
    lambda_vgg: float
    lambda_sobel: float
    lambda_laplacian: float

@dataclass
class ConfigLoss:
    lambda_mse: float
    pixel: ConfigPixelLoss

@dataclass
class ConfigTrain:
    load: ConfigLoad
    loss: ConfigLoss

##### SAMPLING #####

@dataclass
class ConfigSampling:
    function: str
    ddim_timesteps: int
    ddim_recompute_epsilon: bool

##### MLP #####

@dataclass
class ConfigMLP:
    time_embedding_dimension: int
    hidden_dimension: int
    output_dimension: int

##### UNet #####

@dataclass
class ConfigLearningRateWarmRestarts:
    steps_before_first_restart: int
    restart_steps_multiplier: float

@dataclass
class ConfigLearningRateDecay:
    enable: bool
    schedule_type: str
    steps_before_decay: int
    decay_steps: int
    minimum_lr: float
    cosine_cycle_length: int
    warm_restarts: ConfigLearningRateWarmRestarts

@dataclass
class ConfigLearningRate:
    base_rate: float
    warm_up: bool
    warm_up_steps: int
    decay: ConfigLearningRateDecay

@dataclass
class ConfigUNetCompile:
    enable: bool
    mode: str

@dataclass
class ConfigUNetOptimizer:
    optimizer_type: str
    betas: list[float]
    fused: bool

@dataclass
class ConfigUNet:
    in_channels: int
    features: int
    n_downs: int
    middle_layer_depth: int
    self_attention: bool
    enable_checkpointing: bool
    optimizer: ConfigUNetOptimizer
    compile: ConfigUNetCompile
    learning_rate: ConfigLearningRate
    
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
    gradient_accumulation_iterations: int
    noise_schedule: ConfigNoiseSchedule
    sampling: ConfigSampling
    mlp: ConfigMLP
    unet: ConfigUNet

##### CAPTIONS #####

@dataclass
class ConfigCaptions:
    use_captions: True
    metadata_file: str
    encoder_model: str
    max_length: int
    freeze_encoder: bool
    use_fixed_captions: bool
    fixed_captions: list[str]
    
##### LATENTS #####

@dataclass
class ConfigLatentCache:
    read_from_cache: bool
    cache_directory: str
    loader_type: str

@dataclass
class ConfigLatents:
    latent_size_divisor: int
    override_latent_size: bool
    latent_size: int
    vae: str
    vae_device: str
    vae_dtype: str
    scaling_factor: float
    cache: ConfigLatentCache


##### SAVING #####

@dataclass
class ConfigSave:
    save_model: bool
    model_save_rate: int
    auto_increment_version: bool
    save_examples: bool
    example_save_rate: int
    num_examples: int
    sample_cfg_scales: list[float]

##### VISUALIZER #####

@dataclass
class ConfigVisdom:
    enable: bool
    average_loss: bool
    env_name: str
    server: str
    port: int
    display_images: bool
    image_size: int
    max_images: int
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
    train: ConfigTrain
    model: ConfigModel
    dataloader: ConfigDataloader
    latents: ConfigLatents
    captions: ConfigCaptions
    save: ConfigSave
    visualizer: ConfigVisualizer

    DEFAULT_FILE = 'defaults/diffusion.json'
    GROUP_SCHEMA = cfgcommon.Config.GROUP_SCHEMA | {
        'train': ConfigTrain,
        'model': ConfigModel,
        'dataloader': ConfigDataloader,
        'latents': ConfigLatents,
        'captions': ConfigCaptions,
        'save': ConfigSave,
        'visualizer': ConfigVisualizer}
    
    def __post_init__(self) -> None:
        # This is hacky and needs to be fixed in the future.
        self.dataloader.load = ConfigDataloaderLoad(
            load_size=self.model.input_size, crop_size=self.model.input_size, 
            input_nc=self.model.unet.in_channels)

