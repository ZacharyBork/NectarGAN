# NectarGAN Docker console renderer

import subprocess
from typing import Callable

import wrapperutils

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
            text: str='', 
            color: str='WHT', 
            nocolor: bool=False,
            line_end: str | None=None
        ) -> None:
        if nocolor: print(text, end=line_end)
        else: print(self.color_text(text, color), end=line_end)

    def println_split(
            self, 
            prefix_text: str='', 
            prefix_color: str='WHT', 
            body_text: str='', 
            body_color: str='WHT',
            line_end: str | None=None
        ) -> None:        
        print(f'{self.color_text(prefix_text, prefix_color)} '
              f'{self.color_text(body_text, body_color)}', end=line_end)        
    
LR = LineRenderer()


class ConsoleRenderer:
    def __init__(self):
        self.current_status = ('Ready...', 'GRN') # Status: (text, color)
        
    def print_welcome(self) -> None:
        welcome_text = f'''Welcome to...
 _______              __               _______ _______ _______ 
|    |  |.-----.----.|  |_.---.-.----.|     __|   _   |    |  |
|       ||  -__|  __||   _|  _  |   _||    |  |       |       |
|__|____||_____|____||____|___._|__|  |_______|___|___|__|____|
---------------------------------------------------(for Docker)'''
        LR.println(welcome_text, 'ORG')

    def print_commands(self) -> None:
        dataset = 'Current: ' + wrapperutils.get_current_dataset()
        dir = 'Current: ' + wrapperutils.get_current_direction()
        LR.println(nocolor=True, text=f'''Common commands:

{LR.color_text('train', 'GRN')}        | Begin training
{LR.color_text('test', 'GRN')}         | Begin testing
{LR.color_text('swapdir', 'GRN')}      | Switch train/test direction ({dir})
{LR.color_text('dataset-set', 'GRN')}  | Set current dataset ({dataset})
{LR.color_text('dataset-info', 'GRN')} | Info about the current dataset
{LR.color_text('config-edit', 'GRN')}  | Edit config file
{LR.color_text('config-print', 'GRN')} | Print current config
{LR.color_text('shell', 'YLW')}        | Exit startup script
{LR.color_text('exit', 'RED')}         | Exit container\n''')
        
    def add_divider(self, prefix: str='', line_end=None) -> None:
        ln = '---------------------------------------------------------------'
        if not prefix == '':
            length = len(prefix) + 1
            LR.println(f'{prefix} {ln[:-length]}\n', 'ORG', line_end=line_end)
        else: LR.println(f'{ln}\n', 'ORG', line_end=line_end)

    def print_status(self) -> None:
        LR.println_split(
            'Status:', 'ORG', 
            self.current_status[0], self.current_status[1])
        self.add_divider()

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

    def show_command_screen(self) -> None:
        self.reset_console()
        self.print_commands()
    
    def set_status(self, text: str, color: str) -> None:
        self.current_status = (text, color)
        self.reset_console()

    def reset_status(self) -> None:
        self.set_status('Ready...', 'GRN')

    def render_progress_bar(
            self,
            current_value: int, 
            total_value: int,
            character: str='#',
            color: str='GRN',
            total_length: int=63
        ) -> None:
        progress = current_value / total_value
        progress_bar = character * round(progress * total_length)
        LR.println(progress_bar, color)

    def render_progress_bar_full(
            self, label: str, current_value: int, total_value: int,
            character: str='#', color: str='GRN', total_length: int=63
        ) -> None:
        self.add_divider(prefix=label, line_end='')
        self.render_progress_bar(
            current_value, total_value, character, color, total_length)
        self.add_divider(line_end='')

RENDERER = ConsoleRenderer()

