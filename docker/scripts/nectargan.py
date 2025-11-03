# NectarGAN Docker home script

import sys
import os
import subprocess
from pathlib import Path
import json

import config_editor
import renderer as R
C = R.COLORS

MOUNT_DIRECTORY = Path('/app/mount')
CONFIG_PATH = Path('/app/mount/docker_nectargan_config.json')
config_editor.CONFIG_PATH = CONFIG_PATH

def validate_default_directories() -> None:
    if not MOUNT_DIRECTORY.exists():
        raise FileNotFoundError(
            'Unable to locate mount directory at:\n'
            '/app/mount\n\nExiting...')
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            'Unable to locate config file at:\n'
            '/app/mount/docker_nectargan_config.json\n\nExiting...')

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
    
def set_dataset(command: list[str]) -> None:
    try: command[1]
    except Exception:
        print(f'{C.RED}ERROR: No name provided.{C.END}')
        return
    try:
        new_dataroot = Path(MOUNT_DIRECTORY, f'input/{command[1]}') 
        assert new_dataroot.exists()
    except AssertionError:
        print(f'{C.RED}ERROR: Unable to locate dataroot directory.{C.END}')
        print(f'Tried path: {new_dataroot.as_posix()}')
        return
    try:
        with open(CONFIG_PATH, 'r') as file:
            json_data = json.loads(file.read())
        json_data['config']['dataloader']['dataroot'] = new_dataroot.as_posix()
        with open(CONFIG_PATH, 'w') as file:
            file.write(json.dumps(json_data, indent=2))
        print(f'\n{C.ORG}Dataset set!{C.END}')
        print(f'{C.GRN}Path: {C.END}{new_dataroot.as_posix()}\n')
    except Exception as e:
        print(f'{C.RED}ERROR: Unable to load config file.{C.END}')
        print(f'Reason:\n{e}')
        return

if __name__ == '__main__':
    validate_docker_environment()
    validate_default_directories()

    while True:
        R.RENDERER.reset_console()
        print('Please enter a command...')
        command = input(f'{C.ORG}NectarGAN ->{C.END} ').split()
        match command[0].strip().casefold():
            case 'train': 
                proc = subprocess.Popen([
                    'python', '-m', 'nectargan.start.training.paired', '-f', CONFIG_PATH.as_posix()])
                proc.wait()
            case 'test': R.RENDERER.current_status = (C.RED, 'Not yet implemented...')
            case 'dataset-set': set_dataset(command)
            case 'config-edit': 
                R.RENDERER.current_status = (C.GRN, 'Editing config file...')
                config_editor.edit_config_file()
            case 'config-print': 
                R.RENDERER.current_status = (C.GRN, 'Displaying config file...')
                R.RENDERER.clear_console()
                with open(CONFIG_PATH, 'r') as file:
                    data = json.load(file)
                print(f'\n{json.dumps(data, indent=4)}\n')
                input(f'{C.GRN}Press Enter to continue...{C.END}')
                print()
            case 'help': R.RENDERER.current_status = (C.RED, 'Not yet implemented...')
            case 'shell':
                os.chdir('/app')
                os.execv('/bin/sh', ['/bin/sh'])
            case 'exit': sys.exit(0)
            case _: R.RENDERER.current_status = (C.RED, f'\nInvalid command: {command}') 
        