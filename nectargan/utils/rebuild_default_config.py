'''This file serves as a backup definition of the default config.

It also provides a small script to rebuild the default config file in it's 
original location from the `DEFINITION` in this file, just in case it gets lost
or deleted or otherwise broken beyond repair.

Note that, by default, the script in this file will not allow overwriting of an
existing default config if it finds one at `nectargan/config/default.config`.
However, the `-O` (or `--overwrite_existing`) can be used when running the file
to allow it to overwrite the existing default config, effectively restoring it
to its factory default state.
'''

import json
import requests
import argparse
from pathlib import Path
from typing import Any
from importlib.resources import files

URL = 'https://raw.githubusercontent.com/ZacharyBork/NectarGAN/v0.3.0/nectargan/config/default.json'

def get_definition() -> dict[str, Any]:
	response = requests.get(URL)
	response.raise_for_status()
	return response.json()

DEFINITION = get_definition()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--overwrite_existing', action='store_true',
        help='Allows the script to overwrite an existing default config file.')
    args = parser.parse_args()
    
    config_path = Path(files('nectargan.config').joinpath('default.json'))
    if config_path.exists():
        if not args.overwrite_existing:
            raise FileExistsError(
                f'Default config file already exists: {config_path}\n\n'
				f'To continue, either delete the existing config file, or use '
				f'"-overwrite_existing" to allow overwriting.')
    try:
        with open(config_path, 'w') as file:
            json.dump(DEFINITION, file, indent=4)
    except Exception as e:
        raise RuntimeError('Unable to create default config file.') from e

if __name__ == '__main__':
    main()