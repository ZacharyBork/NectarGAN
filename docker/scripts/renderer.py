# NectarGAN Docker console renderer

import subprocess
from typing import Callable

class LineRenderer:
    def __init__(self) -> None:
        self.COLORS = {
            'WHT': '\33[0m',
            'RED': '\033[31m',
            'GRN': '\033[32m',
            'YLW': '\033[33m',
            'ORG': '\033[38;2;255;165;0m'
        }
        self.FORMAT = { 'END': '\033[0m' }
    
    def validate_color(self, text: str, color: str) -> None:
        if not color in self.COLORS.keys():
            raise ValueError(
                f'{self.COLORS['RED']}'
                f'Invalid color provided: "{color}" for line:\n\n{text}'
                f'{self.FORMAT['END']}')
        
    def color_text(self, text: str, color: str) -> str:
        self.validate_color(text, color)
        return f'{self.COLORS[color]}{text}{self.FORMAT['END']}'

    def println(
            self, 
            text: str, 
            color: str='WHT', 
            nocolor: bool=False
        ) -> None:
        if nocolor: print(text)
        else: print(self.color_text(text, color))

    def println_split(
            self, 
            prefix_text: str='', 
            prefix_color: str='WHT', 
            body_text: str='', 
            body_color: str='WHT'
        ) -> None:        
        print(f'{self.color_text(prefix_text, prefix_color)} '
              f'{self.color_text(body_text, body_color)}')        
    
LR = LineRenderer()


class ConsoleRenderer:
    def __init__(self):
        self.first_run = True
        self.current_status = ('Ready...', 'GRN') # Status: (text, color)
        
    def print_welcome(self) -> None:
        prefix = 'Welcome to...' if self.first_run else ''
        welcome_text = f'''{prefix}
 _______              __               _______ _______ _______ 
|    |  |.-----.----.|  |_.---.-.----.|     __|   _   |    |  |
|       ||  -__|  __||   _|  _  |   _||    |  |       |       |
|__|____||_____|____||____|___._|__|  |_______|___|___|__|____|
---------------------------------------------------(for Docker)'''
        LR.println(welcome_text, 'ORG')

    def print_commands(self) -> None:
        LR.println(nocolor=True, text=f'''Common commands:

{LR.color_text('train', 'GRN')}        | Begin training
{LR.color_text('test', 'GRN')}         | Begin testing
{LR.color_text('dataset-set', 'GRN')}  | Set current dataset
{LR.color_text('config-edit', 'GRN')}  | Edit config file
{LR.color_text('config-print', 'GRN')} | Print current config
{LR.color_text('help', 'GRN')}         | See all commands
{LR.color_text('shell', 'YLW')}        | Exit startup script
{LR.color_text('exit', 'RED')}         | Exit container\n''')
        
    def print_status(self) -> None:
        ln = '---------------------------------------------------------------'
        LR.println_split(
            'Status:', 'ORG', 
            self.current_status[0], self.current_status[1])
        LR.println(f'{ln}\n', 'ORG')

    def clear_console(self) -> None:
        subprocess.run(["clear"])

    def reset_console(
            self, 
            inject_header: Callable[[], None] | None=None,
            no_status: bool=False
        ) -> None:
        self.clear_console()
        self.print_welcome()
        if not inject_header is None: inject_header()
        if not no_status: self.print_status()
        self.first_run = False

    def show_command_screen(self) -> None:
        self.reset_console()
        self.print_commands()
    
    def set_status(self, text: str, color: str) -> None:
        self.current_status = (text, color)
        self.reset_console()

    def reset_status(self) -> None:
        self.set_status('Ready...', 'GRN')
    
RENDERER = ConsoleRenderer()

