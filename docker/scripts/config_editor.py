# NectarGAN Docker config value editor

import json
from typing import Any
from pathlib import Path

import renderer as R

CONFIG_PATH: Path = None

def render_config_header() -> None:
    R.LR.println('''                                                
 _____         ___ _        _____   _ _ _           
|     |___ ___|  _|_|___   |   __|_| |_| |_ ___ ___ 
|   --| . |   |  _| | . |  |   __| . | |  _| . |  _|
|_____|___|_|_|_| |_|_  |  |_____|___|_|_| |___|_|  
                    |___|                           
---------------------------------------------------------------''', 'ORG')

def find_requested_config_value(
        json_data: dict
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
        try: 
            current_data = json_data
            for key in keys[:-1]:
                current_data = current_data[key]
            final_key = keys[-1]
            value = current_data[final_key]
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
        json_data: dict[Any], 
        keys: list[str], 
        value: str
    ) -> tuple[bool, bool | int | float | str]:
    while True:
        R.RENDERER.reset_console(inject_header=render_config_header)
        R.LR.println_split('Current Value:', 'GRN', value, 'WHT')
        R.LR.println_split('Value Type:', 'GRN', type(value).__name__(), 'WHT')
        R.LR.println('\nPlease enter a new value (or exit to cancel)...')
        new_value = input(R.LR.color_text(f'Value -> ', 'ORG'))
        if new_value.strip().casefold() == 'exit':
            R.RENDERER.set_status(f'Operation canceled. Exiting...', 'RED')
            return (True, '')
        
        match value:
            case bool():
                match new_value.strip().casefold():
                    case 'true': new_value = True
                    case 'false': new_value = False
                    case _:
                        R.RENDERER.set_status(
                            f'Invalid value: "{new_value}" for type (bool)',
                            'RED')
                        continue
            case int():
                try: new_value = int(new_value)
                except ValueError:
                    R.RENDERER.set_status(
                        f'Invalid value: "{new_value}" for type (int)',
                        'RED')
                    continue
            case float():
                try: new_value = float(new_value)
                except ValueError:
                    R.RENDERER.set_status(
                        f'Invalid value: "{new_value}" for type (float)',
                        'RED')
                    continue
            case str(): pass
            case _: 
                R.RENDERER.set_status(
                    f'Unsupported value type: {type(new_value)}',
                    'RED')
                continue

        current_data = json_data
        for key in keys[:-1]:
            current_data = current_data[key]
        current_data[keys[-1]] = new_value
        break
    return (False, new_value)

def edit_config_file() -> None:
    if CONFIG_PATH is None:
        R.RENDERER.set_status(
            'Unable to locate config file! Exiting...', 'RED')
        return
    
    with open(CONFIG_PATH, 'r') as file:
        json_data = json.loads(file.read())
    
    canceled, keys, value, userinput = find_requested_config_value(json_data)
    if canceled: return
    canceled, new_value = update_value(json_data, keys, value)
    if canceled: return
    
    try:
        with open(CONFIG_PATH, 'w') as file:
            file.write(json.dumps(json_data, indent=2))
    except Exception as e:
        R.RENDERER.set_status('Unable to save config file.', 'RED')
        return
    R.LR.println('Update Successful!', 'GRN')
    R.LR.println_split('Variable:', 'GRN', userinput, 'WHT')
    R.LR.println_split('New Value:', 'GRN', new_value, 'WHT')
    
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