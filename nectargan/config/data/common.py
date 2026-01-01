from dataclasses import dataclass

##### COMMONS #####

@dataclass
class ConfigCUDNN:
    benchmark: bool
    deterministic: bool

@dataclass
class ConfigCommon:
    device: str
    gpu_ids: list[int]
    cudnn: ConfigCUDNN
    output_directory: str
    experiment_name: str
    experiment_version: int

##### BASE CONFIG #####

@dataclass
class Config:
    config_type: str
    common: ConfigCommon

    DEFAULT_FILE = None
    GROUP_SCHEMA = { 'common': ConfigCommon }
    


