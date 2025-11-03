import subprocess
from dataclasses import dataclass

@dataclass
class COLORSBASE:
    ORG = '\033[38;2;255;165;0m'
    RED = '\033[31m'
    GRN = '\033[32m'
    YLW = '\033[33m'
    
    END = '\033[0m'
COLORS = COLORSBASE()

class ConsoleRenderer:
    def __init__(self):
        # Status: (color, text)
        self.current_status = (None, '')

    def print_welcome(self) -> None:
        print(f'''{COLORS.ORG}
Welcome to...
 _______              __               _______ _______ _______ 
|    |  |.-----.----.|  |_.---.-.----.|     __|   _   |    |  |
|       ||  -__|  __||   _|  _  |   _||    |  |       |       |
|__|____||_____|____||____|___._|__|  |_______|___|___|__|____|
---------------------------------------------------(for Docker){COLORS.END}''')

    def print_commands(self) -> None:
        print(f'''Common commands:

{COLORS.GRN}train{COLORS.END}        | Begin training
{COLORS.GRN}test{COLORS.END}         | Begin testing
{COLORS.GRN}dataset-set{COLORS.END}  | Set current dataset
{COLORS.GRN}config-edit{COLORS.END}  | Edit config file
{COLORS.GRN}config-print{COLORS.END} | Print current config
{COLORS.GRN}help{COLORS.END}         | See all commands
{COLORS.YLW}shell{COLORS.END}        | Exit startup script
{COLORS.RED}exit{COLORS.END}         | Exit container
''' )
        
    def print_status(self) -> None:
        status = ''
        if not self.current_status[0] is None: status += self.current_status[0]
        status += self.current_status[1]
        if not self.current_status[0] is None: status += COLORS.END
        status_box = f'''{COLORS.ORG}Status:{COLORS.END} {status}
{COLORS.ORG}---------------------------------------------------------------
{COLORS.END}'''
        print(status_box)
        self.current_status = (None, '')

    def clear_console(self) -> None:
        subprocess.run(["clear"])
        self.print_welcome()
        self.print_status()

    def reset_console(self) -> None:
        self.clear_console()
        self.print_commands()
    
RENDERER = ConsoleRenderer()
