# NectarGAN Docker config value editor

import json
from typing import Any
from pathlib import Path

import renderer as R
import wrapperutils

def render_config_header() -> None:
    R.LR.println('''                                                
 _____         ___ _        _____   _ _ _           
|     |___ ___|  _|_|___   |   __|_| |_| |_ ___ ___ 
|   --| . |   |  _| | . |  |   __| . | |  _| . |  _|
|_____|___|_|_|_| |_|_  |  |_____|___|_|_| |___|_|  
                    |___|                           
---------------------------------------------------------------''', 'ORG')

def find_requested_config_value(
        json_data: dict[str, Any]
    ) -> tuple[bool, list[str], str, str]:
    key_prompt = '''
Enter the set of config keys, separated by ".", pointing to the 
config value you would like to edit, or "exit" to cancel.

e.g. config.train.generator.block_type
'''
    while True:
        R.RENDERER.reset_console(inject_header=render_config_header)
        R.LR.println(key_prompt)
        userinput = input(R.LR.color_text(f'Keys -> ', 'ORG'))
        keys = userinput.split('.')
        if keys[0].strip().casefold() == 'exit':
            R.RENDERER.set_status('Operation canceled. Exiting...', 'RED')
            return (True, [], '', '')
        if len(keys) == 0:
            R.RENDERER.set_status('No keys provided!', 'RED')
            continue
        try: value = wrapperutils.get_config_value(keys)
        except KeyError:
            R.RENDERER.set_status(
                f'Unable to located config value with keys: {keys}', 'RED')
            continue

        R.LR.println_split('Keys:', 'GRN', userinput, 'WHT')
        if not isinstance(value, (bool, int, float, str)):
            R.RENDERER.set_status(
                f'Type not supported: {type(value)} | '
                f'Supported: (bool, int, float, str)',
                'RED')
            continue
        break
    return (False, keys, value, userinput)

def update_value(
        keys: list[str], 
        value: str
    ) -> tuple[bool, Any]:
    while True:
        R.RENDERER.reset_console(inject_header=render_config_header)
        R.LR.println_split('Current Value:', 'GRN', value, 'WHT')
        R.LR.println_split('Value Type:', 'GRN', type(value), 'WHT')
        R.LR.println('\nPlease enter a new value (or exit to cancel)...')
        new_value = input(R.LR.color_text(f'Value -> ', 'ORG'))
        if new_value.strip().casefold() == 'exit':
            R.RENDERER.set_status(f'Operation canceled. Exiting...', 'RED')
            return (True, '')
        
        try: wrapperutils.set_config_value(keys, new_value)
        except ValueError:
            R.RENDERER.set_status(
                f'Unsupported value type: {type(new_value)}', 'RED')
            continue
        break
    return (False, new_value)

def edit_config_file() -> None:
    R.RENDERER.set_status('Editing config file...', 'GRN')

    json_data = wrapperutils.read_config()
    canceled, keys, value, userinput = find_requested_config_value(json_data)
    if canceled: return
    canceled, new_value = update_value(keys, value)
    if canceled: return
    
    R.RENDERER.reset_console(inject_header=render_config_header)
    R.LR.println('Update Successful!', 'GRN')
    R.LR.println_split('Variable:', 'GRN', userinput, 'WHT')
    R.LR.println_split('New Value:', 'GRN', new_value, 'WHT')
    input(R.LR.color_text('\nPress enter to confirm...', 'ORG'))
    
    while True:
        R.RENDERER.reset_console(inject_header=render_config_header)
        R.LR.println('Continue editing? (y | n)')
        choice = input(R.LR.color_text(f'Continue? -> ', 'ORG'))
        match choice.strip().casefold():
            case 'y':
                edit_config_file()
            case 'n':
                R.RENDERER.set_status('Finished! Exiting...', 'GRN')
                return
            case _:
                R.RENDERER.set_status(f'Input not valid: {choice}', 'RED')

                
