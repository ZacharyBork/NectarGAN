# General utility scripts for NectarGAN Docker environment

from pathlib import Path

def validate_docker_environment() -> None:
    '''Ensures the script is being run from within a Docker container.
    
    Raises:
        OSError : If unable to find "/.dockerenv" in host filesystem.
    '''
    try: assert Path('/.dockerenv').exists()
    except AssertionError:
        raise (OSError(
            f'This script is intended to be run from within a containerized '
            f'environment, and will not function correctly otherwise.\n\n'
            f'Aborting...'))
    