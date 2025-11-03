# NectarGAN Docker home script

import sys
import os
import subprocess
from pathlib import Path
import json

MOUNT_DIRECTORY = Path('/app/mount')
CONFIG_PATH = Path('/app/mount/docker_nectargan_config.json')

welcome_message = f'''\033[38;2;255;165;0m
Welcome to...
 _______              __               _______ _______ _______ 
|    |  |.-----.----.|  |_.---.-.----.|     __|   _   |    |  |
|       ||  -__|  __||   _|  _  |   _||    |  |       |       |
|__|____||_____|____||____|___._|__|  |_______|___|___|__|____|
---------------------------------------------------(for Docker)\033[0m'''
commands_message = f'''Common commands:

\033[32mtrain\033[0m        | Begin training
\033[32mtest\033[0m         | Begin testing
\033[32mdataset-set\033[0m  | Set current dataset
\033[32mconfig-edit\033[0m  | Edit config file
\033[32mconfig-print\033[0m | Print current config
\033[32mhelp\033[0m         | See all commands
\033[33mshell\033[0m        | Exit startup script
\033[31mexit\033[0m         | Exit container
'''

def validate_default_directories() -> None:
    assert MOUNT_DIRECTORY.exists()
    assert CONFIG_PATH.exists()

def validate_docker_environment() -> None:
    '''Ensures the script is being run from within a Docker container.
    
    Raises:
        OSError : If unable to find '/.dockerenv' in host filesystem.
    '''
    try: assert Path('/.dockerenv').exists()
    except AssertionError:
        raise (OSError(
            f'This script is intended to be run from within a containerized '
            f'environment, and will not function correctly otherwise.\n\n'
            f'Aborting...'))  
    
def set_dataset(command: list[str]) -> None:
    try: command[1]
    except AssertionError:
        print('\033[31mERROR: No name provided.\033[0m')
        return
    try:
        new_dataroot = Path(MOUNT_DIRECTORY, f'input/{command[1]}') 
        assert new_dataroot.exists()
    except AssertionError:
        print('\033[31mERROR: Unable to locate dataroot directory.\033[0m')
        print(f'Tried path: {new_dataroot.as_posix()}')
        return
    try:
        with open(CONFIG_PATH, 'r') as file:
            json_data = json.loads(file.read())
        json_data['config']['dataloader']['dataroot'] = new_dataroot.as_posix()
        with open(CONFIG_PATH, 'w') as file:
            file.write(json.dumps(json_data, indent=2))
        print('\n\033[38;2;255;165;0mDataset set!\033[0m')
        print(f'\033[32mPath: \033[0m{new_dataroot.as_posix()}\n')
    except Exception as e:
        print('\033[31mERROR: Unable to load config file.\033[0m')
        print(f'Reason:\n{e}')
        return

if __name__ == '__main__':
    validate_docker_environment()
    print(welcome_message)
    while True:
        print(commands_message)
        print('Please enter a command...')
        command = input('\033[38;2;255;165;0mNectarGAN -> \033[0m').split(' ')
        match command[0].lower():
            case 'train': 
                proc = subprocess.Popen([
                    'python', '-m', 'nectargan.start.training.paired', '-f', CONFIG_PATH.as_posix()])
                proc.wait()
            case 'test': print('Not yet implemented.')
            case 'dataset-set': set_dataset(command)
            case 'config-edit': print('Not yet implemented.')
            case 'config-print': subprocess.Popen(['cat', CONFIG_PATH.as_posix()])
            case 'shell':
                os.chdir('/app')
                os.execv('/bin/sh', ['/bin/sh'])
            case 'exit': sys.exit(0)
            case _: print(f'\n\033[31mInvalid command:\033[0m {command}')
