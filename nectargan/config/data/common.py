from dataclasses import dataclass

@dataclass
class Config:
    config_type: str

@dataclass
class ConfigCommon:
    device: str
    gpu_ids: list[int]
    output_directory: str
    experiment_name: str
    experiment_version: int

@dataclass
class ConfigLearningRate:
    epochs: int
    epochs_decay: int
    initial: float
    target: float

@dataclass
class ConfigDataloaderLoad:
    load_size: int
    crop_size: int
    input_nc: int

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
class ConfigVisdom:
    enable: bool
    env_name: str
    server: str
    port: int
    image_size: int
    update_frequency: int
    

