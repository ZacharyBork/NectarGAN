import os
import json
from pathlib import Path
from typing import Any

import renderer as R

# These defaults should probably be moved to env at some point
# Default config file path
CONFIG_PATH: Path = '/app/mount/docker_nectargan_config.json'
# Default mount directory
MOUNT_DIRECTORY = '/app/mount'

##### Docker #####

def validate_docker_environment() -> None:
    '''Ensures the script is being run from within a Docker container.
    
    Raises:
        OSError : If unable to find '/.dockerenv' in host filesystem.
    '''
    if not Path('/.dockerenv').exists():
        raise OSError(
            f'This script is intended to be run from within a containerized '
            f'environment, and will not function correctly otherwise.\n\n'
            f'Aborting...') 
    
def exit_to_shell() -> None:
    R.RENDERER.set_status('Exiting to shell...', 'YLW')
    R.LR.println(
        f'This will exit the NectarGAN CLI wrapper to the '
        f'container\'s shell environment.\n\n'
        f'Enter (Y | y) to confirm.', 'YLW')
    confirm = input(R.LR.color_text('Confirm -> ', 'ORG'))
    if confirm.strip().casefold() == 'y':
        R.RENDERER.reset_console(no_status=True)
        R.LR.println_split(
            '\nExiting...', 'ORG',
            '(Hint: Use the "nectargan" command to restart the wrapper.)\n', 
            'GRN')
        os.chdir('/app')
        os.execv('/bin/sh', ['/bin/sh'])
    else: R.RENDERER.set_status('Aborted...', 'RED')

##### MOUNT DIRECTORY #####

def get_mount_directory() -> Path:
    mount_directory = Path(MOUNT_DIRECTORY)
    if not mount_directory.exists():
        raise FileNotFoundError(
            'Unable to locate mount directory at:\n'
            '/app/mount\n\nExiting...')
    return mount_directory

##### CONFIG UTILS #####

def get_config_path() -> Path:
    config_path = Path(CONFIG_PATH)
    if not config_path.exists():
        raise FileNotFoundError(
            'Unable to locate config file at:\n'
            '/app/mount/docker_nectargan_config.json\n\nExiting...')
    return config_path

def read_config() -> dict[str, Any]:
    if CONFIG_PATH is None: raise ValueError('Config file path not set!')
    try:
        with open(CONFIG_PATH, 'r') as config_file:
            config_data = json.loads(config_file.read())
    except Exception as e:
        raise FileNotFoundError(
            f'Unable to locate config file at path: {CONFIG_PATH.as_posix()}')
    return config_data

def write_config(config_data: dict[str, Any]) -> None:
    if CONFIG_PATH is None: raise ValueError('Config file path not set!')
    try:
        with open(CONFIG_PATH, 'w') as config_file:
            config_file.write(json.dumps(config_data, indent=2))
    except Exception as e:
        raise FileNotFoundError(
            f'Unable to locate config file at path: {CONFIG_PATH.as_posix()}')
    
def print_config_data() -> None:
    R.RENDERER.set_status('Displaying config file...', 'GRN')
    data = read_config()
    R.LR.println(f'\n{json.dumps(data, indent=2)}\n')
    input(R.LR.color_text(f'Press Enter to continue...', 'GRN'))
    R.LR.println()
    R.RENDERER.reset_status()

def set_value(keys: dict[str, Any], value: Any) -> dict[str, Any]:
    config_data = read_config()
    current_data = config_data
    for key in keys[:-1]:
        current_data = current_data[key]
    current_data[keys[-1]] = value
    return config_data

def get_config_value(keys: list[str]) -> Any:
    if len(keys) == 0: raise ValueError('No keys provided for config search!')
    config_data = read_config()
    value = config_data
    for key in keys:
        try: value = value[key]
        except: raise KeyError(f'Invalid config search key: {key}')
    return value

def set_config_value(keys: list[str], value: Any) -> Any:
    if len(keys) == 0: raise ValueError('No keys provided for config search!')
    current_value = get_config_value(keys)
    if not type(current_value) == type(value):
        raise ValueError(
            f'Invalid value: {value} for type ({type(current_value)})')
    updated_data = set_value(keys, value)
    write_config(updated_data)

##### DATASET #####

def get_current_dataset() -> str:
    dataset = Path(get_config_value(
        ['config', 'dataloader', 'dataroot']))
    if dataset is None or dataset == Path('/app/mount/input'): 
        return R.LR.color_text('No dataset set!', 'RED')
    
    for i in ['train', 'val']:
        subdirectory = Path(dataset, i)
        if not subdirectory.exists():
            return R.LR.color_text(f'Dataset missing split "{i}"!', 'RED')
        
    return R.LR.color_text(dataset.name, 'GRN') 

def set_dataset(command: list[str]) -> None:
    try: command[1]
    except Exception:
        R.RENDERER.set_status('ERROR: No name provided.', 'RED')
        return
    new_dataroot = Path(
        get_mount_directory(), f'input/{command[1]}') 
    if not new_dataroot.exists():
        R.RENDERER.set_status(
            f'ERROR: Unable to locate dataroot directory at '
            f'path: {new_dataroot.as_posix()}', 'RED')
        return
    try:
        set_config_value(
            keys=['config', 'dataloader', 'dataroot'],
            value=new_dataroot.as_posix())
        R.RENDERER.set_status(
            f'Dataset set! Path: {new_dataroot.as_posix()}', 'GRN')
    except Exception:
        R.RENDERER.set_status(f'ERROR: Unable to load config file.', 'RED')
        return
    
def get_dataset_info() -> None:
    dataset = Path(get_config_value(
        ['config', 'dataloader', 'dataroot']))
    if not dataset.exists():
        R.RENDERER.set_status(
            f'Unable to locate dataset at path: {dataset.as_posix()}', 'RED')
        return
    R.RENDERER.set_status('Checking dataset...', 'GRN')
    R.LR.println_split('Path:', 'ORG', dataset.as_posix(), 'GRN')
    R.LR.println()
    R.RENDERER.add_divider()
    R.LR.println('File Counts:', 'ORG')

    counts = { 'test': 0, 'train': 0, 'val': 0 }
    for i in ['test', 'train', 'val']:
        subdirectory = Path(dataset, i)
        if not subdirectory.exists(): 
            R.LR.println_split(
                f'- {i.capitalize()}:', 'ORG', 'Not found.', 'RED')
        else: 
            num_files = len(list(subdirectory.iterdir()))
            if num_files > 0: 
                counts[i] = num_files
                color = 'GRN'
            else: color = 'RED'
        R.LR.println_split(
            f'- {i.capitalize()}:', 'ORG', str(num_files), color)
    R.LR.println()
    R.RENDERER.add_divider()
    total_files = 0
    for value in counts.values(): total_files += value
    R.LR.println_split('Total files:', 'ORG', str(total_files), 'GRN')
    R.LR.println('Splits:', 'ORG')
    for i in ['test', 'train', 'val']:
        percent = float(counts[i]) / max(0.0001, float(total_files)) * 100.0
        percent = round(percent, 2)
        color = 'GRN' if percent > 0.0 else 'RED'
        R.LR.println_split(
            f'- {i.capitalize()}:', 'ORG', f'{str(percent)}%', color)
    
    R.LR.println()
    R.RENDERER.add_divider()
    input(R.LR.color_text('Press enter to continue...', 'ORG'))

    
##### DIRECTION #####

def _get_direction() -> tuple[bool, str]:
    was_valid = True
    direction = get_config_value(['config', 'dataloader', 'direction'])
    if not direction in ['AtoB', 'BtoA']:
        R.RENDERER.set_status(
            f'Invalid direction "{direction}"! Resetting to default...', 'RED')
        direction = 'AtoB'
        was_valid = False
    return (was_valid, direction)

def get_current_direction() -> str:
    return R.LR.color_text(_get_direction()[1], 'GRN')

def swap_direction() -> None:
    was_valid, direction = _get_direction()
    if was_valid: new_direction = 'BtoA' if 'AtoB' == direction else 'AtoB'
    set_config_value(['config', 'dataloader', 'direction'], new_direction)
    R.RENDERER.set_status(
        f'Direction changed. New direction: {new_direction}', 'GRN')


