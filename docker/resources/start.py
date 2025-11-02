# The purpose of this file is to hold open the new container for CLI input and route 
# based on input to the various entrypoint scripts.

import subprocess
import signal

import utils

if __name__ == "__main__":
    utils.validate_docker_environment()
    while True:
        print(
            f'Welcome to NectarGAN (Docker)!\n\n'
            f'Common commands:\n'
            f'Begin training: train\n'
            f'Edit config file: edit-config'
            f'Quit NectarGAN: exit\n\n'
            f'Hint: You can use the "help" command to access the full list '
            f'of available commands.'
        )
        print('Please enter a command...')
        command = input('NectarGAN -> :')
        match command.lower():
            case 'train': 
                if hasattr(signal, "SIGINT"):
                    old = signal.signal(signal.SIGINT, signal.SIG_IGN)
                try:
                    proc = subprocess.Popen(
                        ["python", "-m", "nectargan.start.training.paired", "-f", "/app/mount/train_config.json"])
                finally:
                    if hasattr(signal, "SIGINT"):
                        signal.signal(signal.SIGINT, old)
                proc.wait()
            case 'edit-config': print('Not yet implemented.')
            case 'exit': exit(0)
            case _: print(f'Invalid command: {command}')