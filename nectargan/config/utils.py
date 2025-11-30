from os import PathLike

from nectargan.config import ConfigManager, GANConfig, DiffusionConfig

def config_from_file(config_file: PathLike) -> GANConfig | DiffusionConfig:
    manager = ConfigManager(config_file)
    data = manager.data
    ctype = data.config_type
    match ctype:
        case 'pix2pix'  : config: GANConfig = data
        case 'diffusion': config: DiffusionConfig = data
        case _: raise ValueError(f'Unrecognized config type: {ctype}')
    return config

