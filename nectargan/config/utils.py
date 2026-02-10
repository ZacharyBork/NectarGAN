import json
from os import PathLike
from pathlib import Path
from typing import Any

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

def get_default_config(
        config_type: str, 
        as_json: bool=False
    ) -> GANConfig | DiffusionConfig | dict[str, Any]:
    root = Path(__file__).parent.resolve()
    match config_type:
        case 'pix2pix'  : config_path = Path(root, 'default.json')
        case 'diffusion': config_path = Path(root, 'defaults/diffusion.json')
        case _: raise ValueError(f'Unrecognized config type: {config_type}')
    if not as_json: return config_from_file(config_path)
    else:
        with open(config_path, 'r') as file:
            config_data = json.loads(file.read())
        return config_data
