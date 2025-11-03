import json
from typing import Any

import renderer as R
C = R.COLORS

CONFIG_PATH = None

def find_requested_config_value(json_data: dict) -> tuple[bool, list[str], str, str]:
    key_prompt = '''
Enter the set of config keys, separated by ".", pointing to the 
config value you would like to edit, or "exit" to cancel.

e.g. config.train.generator.block_type
'''
    while True:
        R.RENDERER.clear_console()
        print(key_prompt)
        userinput = input(f'{C.ORG}Keys -> {C.END}')
        keys = userinput.split('.')
        if keys[0].strip().casefold() == 'exit':
            R.RENDERER.current_status = (C.RED, 'Operation canceled. Exiting...')
            return (True, [], '', '')
        if len(keys) == 0:
            R.RENDERER.current_status = (C.RED, 'No keys provided!')
            continue
        try: 
            current_data = json_data
            for key in keys[:-1]:
                current_data = current_data[key]
            final_key = keys[-1]
            value = current_data[final_key]
        except KeyError:
            R.RENDERER.current_status = (C.RED, f'Unable to located config value with keys: {keys}')
            continue

        print(f'{C.GRN}Keys:{C.END} {userinput}')
        if not isinstance(value, (bool, int, float, str)):
            R.RENDERER.current_status = (
                C.RED, f'Type not supported: {type(value)} | Supported: (bool, int, float, str)')
            continue
        
        break
    return (False, keys, value, userinput)

def update_value(json_data: dict[Any], keys: list[str], value: str) -> tuple[bool, bool | int | float | str]:
    while True:
        R.RENDERER.clear_console()
        print(f'{C.GRN}Current Value:{C.END} {value}')
        print(f'{C.GRN}Value Type:{C.END} {type(value)}')
        print('\nPlease enter a new value (or exit to cancel)...')
        new_value = input(f'{C.ORG}Value -> {C.END}')
        if new_value.strip().casefold() == 'exit':
            R.RENDERER.current_status = (C.RED, f'Operation canceled. Exiting...')
            canceled = True
            return (True, '')
        
        match value:
            case bool():
                match new_value.strip().casefold():
                    case 'true': new_value = True
                    case 'false': new_value = False
                    case _:
                        R.RENDERER.current_status = (
                            C.RED, f'Invalid value: "{new_value}" for type (bool)')
                        continue
            case int():
                try: new_value = int(new_value)
                except ValueError:
                    R.RENDERER.current_status = (
                        C.RED, f'Invalid value: "{new_value}" for type (int)')
                    continue
            case float():
                try: new_value = float(new_value)
                except ValueError:
                    R.RENDERER.current_status = (
                        C.RED, f'Invalid value: "{new_value}" for type (float)')
                    continue
            case str(): pass
            case _: 
                R.RENDERER.current_status = (
                    C.RED, f'Unsupported value type: {type(new_value)}')
                continue

        current_data = json_data
        for key in keys[:-1]:
            current_data = current_data[key]
        current_data[keys[-1]] = new_value
        break
    return (False, new_value)

def edit_config_file() -> None:
    if CONFIG_PATH is None:
        R.RENDERER.current_status = (
            C.RED, 'Unable to locate config file! Exiting...')
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
        R.RENDERER.current_status = (C.RED, 'Unable to save config file.')
        return
    print(f'{C.GRN}Update Successful!{C.END}')
    print(f'{C.GRN}Variable:{C.END} {userinput}')
    print(f'{C.GRN}New Value:{C.END} {new_value}\n')
    
    while True:
        R.RENDERER.clear_console()
        print('Continue editing? (y | n)')
        choice = input(f'{C.ORG}Continue? -> {C.END}')
        match choice.strip().casefold():
            case 'y':
                edit_config_file()
            case 'n':
                R.RENDERER.current_status = (C.GRN, 'Finished! Exiting...')
                return
            case _:
                R.RENDERER.current_status = (C.RED, f'Input not valid: {choice}')