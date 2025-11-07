# NectarGAN Docker home script

import sys
import time

import renderer as R
import wrapperutils
import config_editor
import test_model
import train_model
    
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
        case 'train': train_model.begin_training()
        case 'test': test_model.run_model_test()
        case 'dataset-set': wrapperutils.set_dataset(command)
        case 'config-edit': config_editor.edit_config_file()
        case 'config-print': wrapperutils.print_config_data()
        case 'help': R.RENDERER.set_status('Not yet implemented...', 'RED')
        case 'shell': wrapperutils.exit_to_shell()
        case 'exit': sys.exit(0)
        case _: R.RENDERER.set_status(f'\nInvalid command: {command}', 'RED')

if __name__ == '__main__':
    wrapperutils.validate_docker_environment()    
    play_welcome_sequence()
    while True: handle_input() 
        