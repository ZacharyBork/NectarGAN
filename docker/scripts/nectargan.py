# NectarGAN Docker home script

import sys
import os
import time
import subprocess
from pathlib import Path
import json

import renderer as R
import config_editor

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
    
def begin_training() -> None:
    proc = subprocess.Popen([
        'python', '-m', 
        'nectargan.start.training.paired', 
        '-f', CONFIG_PATH.as_posix()])
    proc.wait()

def set_dataset(command: list[str]) -> None:
    try: command[1]
    except Exception:
        R.RENDERER.set_status('ERROR: No name provided.', 'RED')
        return
    new_dataroot = Path(MOUNT_DIRECTORY, f'input/{command[1]}') 
    if not new_dataroot.exists():
        R.RENDERER.set_status(
            f'ERROR: Unable to locate dataroot directory at '
            f'path: {new_dataroot.as_posix()}', 'RED')
        return
    try:
        with open(CONFIG_PATH, 'r') as file:
            json_data = json.loads(file.read())
        json_data['config']['dataloader']['dataroot'] = new_dataroot.as_posix()
        with open(CONFIG_PATH, 'w') as file:
            file.write(json.dumps(json_data, indent=2))

        R.RENDERER.set_status(
            f'\nDataset set! Path: {new_dataroot.as_posix()}', 'GRN')
    except Exception as e:
        R.RENDERER.set_status(f'ERROR: Unable to load config file.', 'RED')
        return
    
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

def play_welcome_sequence() -> None:
    welcome_message = 'Welcome to...'
    current_message = ''
    for i in welcome_message:
        time.sleep(0.1)
        R.RENDERER.clear_console()
        current_message += i
        R.LR.println(current_message, 'GRN')
    time.sleep(0.5)
    R.RENDERER.clear_console()

def handle_input() -> None:
    R.RENDERER.show_command_screen()
    R.LR.println('Please enter a command...')
    command = input(R.LR.color_text('NectarGAN -> ', 'ORG')).split()
    match command[0].strip().casefold():
        case 'train': begin_training()
        case 'test': R.RENDERER.set_status('Not yet implemented...', 'RED')
        case 'dataset-set': set_dataset(command)
        case 'config-edit': 
            R.RENDERER.set_status('Editing config file...', 'GRN')
            config_editor.edit_config_file()
        case 'config-print': 
            R.RENDERER.set_status('Displaying config file...', 'GRN')
            with open(CONFIG_PATH, 'r') as file:
                data = json.load(file)
            print(f'\n{json.dumps(data, indent=4)}\n')
            input(R.LR.color_text(f'Press Enter to continue...', 'GRN'))
            print()
            R.RENDERER.reset_status()
        case 'help': R.RENDERER.set_status('Not yet implemented...', 'RED')
        case 'shell': exit_to_shell()
        case 'exit': sys.exit(0)
        case _: 
            R.RENDERER.set_status(f'\nInvalid command: {command}', 'RED')

if __name__ == '__main__':
    validate_docker_environment()
    validate_default_directories()
    
    play_welcome_sequence()
    while True: handle_input() 
        