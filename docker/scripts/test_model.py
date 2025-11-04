import subprocess
from pathlib import Path

import renderer as R

MOUNT_DIRECTORY: Path = None
CONFIG_PATH: Path = None

def find_experiment_directories(output_directory: Path) -> list[Path]:
    return [i for i in output_directory.iterdir() if i.is_dir()]

def _get_experiment_directory() -> Path:
    output_directory = Path(MOUNT_DIRECTORY, 'output')
    if not output_directory.exists(): raise FileNotFoundError()

    experiment_dirs = find_experiment_directories(output_directory)
    if len(experiment_dirs) == 0: raise RuntimeError()
    names = [i.name for i in experiment_dirs]

    while True:
        R.RENDERER.reset_console()
        msg = (
            f'Enter the name of the experiment directory containing '
            f'the model\ncheckpoint file you would like to test, or '
            f'"exit" to quit.\n\n'
            f'Found experiment directories '
            f'({R.LR.color_text(str(len(experiment_dirs)), 'GRN')}):')
        R.LR.println(msg)
        for name in names: R.LR.println(name, 'GRN')
        
        R.LR.println('\nPlease enter an experiment name...')
        selected = input(R.LR.color_text('Experiment Name -> ', 'ORG'))
        if selected == 'exit': raise InterruptedError()
        if not selected in names:
            R.RENDERER.set_status(f'Name not valid: {selected}', 'RED')
            continue
        break
    
    return Path(output_directory, selected)

def get_experiment_directory() -> tuple[bool, Path]:
    try: experiment_directory = _get_experiment_directory()
    except FileNotFoundError:
        R.RENDERER.set_status(
            'Unable to locate mounted output directory!', 'RED')
        return (False, None)
    except RuntimeError:
        R.RENDERER.set_status(
            'No experiments found in output directory!', 'RED')
        return (False, None)
    except InterruptedError:
        R.RENDERER.set_status('Exiting...', 'GRN')
        return (False, None)
    except Exception:
        R.RENDERER.set_status(
            'Encountered an unknown error while setting value.', 
            'RED')
        return (False, None)
    return (True, experiment_directory)

def get_load_epoch(experiment_directory: Path) -> tuple[bool, int]:
    checkpoints = [i for i in experiment_directory.glob('*netG.pth*')]
    if len(checkpoints) == 0:
        R.RENDERER.set_status(
            'No checkpoints found in experiment directory!', 'RED')
        return (False, -1)
    
    epochs = []
    while True:
        R.RENDERER.reset_console()
        msg = (
            f'Please enter the checkpoint epoch you would like to load for\n'
            f'testing, or "exit" to quit.\n\n'
            f'Epochs found in experiment directory:')
        R.LR.println(msg)
        for i in checkpoints: 
            epoch = ''.join(j for j in i.stem if j.isdigit())
            R.LR.println_split(f'{i.name}:', 'GRN', epoch, 'ORG')
            epochs.append(epoch)
        
        R.LR.println('\nPlease enter an epoch...')
        selected = input(R.LR.color_text('Load Epoch -> ', 'ORG'))
        if selected == 'exit':
            R.RENDERER.set_status('Exiting...', 'GRN')
            return (False, 0)

        if not selected in epochs:
            R.RENDERER.set_status(f'Epoch not valid: {selected}', 'RED')
            continue
        break
    
    load_epoch = int(''.join(j for j in i.stem if j.isdigit()))
    return (True, load_epoch)

def get_dataroot() -> tuple[bool, Path, Path]:
    input_directory = Path(MOUNT_DIRECTORY, 'input')
    if not input_directory.exists():
        R.RENDERER.set_status(
            'Unable to locate mounted input directory!', 'RED')
        return (False, None)

    dataset_directories = [i for i in input_directory.iterdir() if i.is_dir()]
    if len(dataset_directories) == 0: raise RuntimeError()
    names = [i.name for i in dataset_directories]

    while True:
        R.RENDERER.reset_console()
        msg = (
            f'Enter the name of the dataset directory for testing, or "exit" '
            f'to quit.\n\n'
            f'Dataset directory must contain a subdirectory called "test" '
            f'which\ncontains the images you\'d like to use for testing.\n\n'
            f'Found experiment directories'
            f'({R.LR.color_text(str(len(dataset_directories)), 'GRN')}):')
        R.LR.println(msg)
        for name in names: R.LR.println(name, 'GRN')
        
        R.LR.println('\nPlease enter an dataset name...')
        selected = input(R.LR.color_text('Dataset Name -> ', 'ORG'))
        if selected == 'exit': 
            R.RENDERER.set_status('Exiting...', 'GRN')
            return (False, None, None)
        if not selected in names:
            R.RENDERER.set_status(f'Name not valid: {selected}', 'RED')
            continue
        
        dataroot = Path(input_directory, selected)
        test_dir = Path(dataroot, 'test')
        if not test_dir.exists():
            R.RENDERER.set_status('Unable to locate test directory...', 'RED')
            return (False, None, None)
        if len(list(test_dir.iterdir())) == 0:
            R.RENDERER.set_status(f'No files found in test directory!', 'RED')
            continue
        break
    return (True, dataroot, test_dir)

def get_test_iterations(test_dir: Path) -> tuple[bool, int]:
    filecount = len(list(test_dir.iterdir()))

    while True:
        R.RENDERER.reset_console()
        msg = (
            f'Please enter a value for the number of test iterations to '
            f'conduct,\nor "exit" to quit.\n\n'
            f'This value will determine how many images from the "test" '
            f'directory\nthe model\'s inference will be run on. It will be '
            f'capped at the count\nof images in the "test" directory.\n\n')
        R.LR.println(msg)
        R.LR.println_split(
            f'Test images directory:', 'GRN', test_dir.as_posix(), 'WHT')
        R.LR.println_split(f'Files found:', 'GRN', str(filecount), 'WHT')

        R.LR.println('\nPlease enter a value for test iterations...')
        selected = input(R.LR.color_text('Iterations -> ', 'ORG'))
        if selected == 'exit': 
            R.RENDERER.set_status('Exiting...', 'GRN')
            return (False, None)
        try: value = int(selected)
        except Exception:
            R.RENDERER.set_status(
                f'Invalid value! Value must be an integer', 'RED')
            continue
        if not value > 0:
            R.RENDERER.set_status(
                f'Invalid value! Value must be greater than 1!', 'RED')
            continue
        if value > filecount:
            R.LR.println(
                nocolor=True, 
                text=f'Given value ({R.LR.color_text(str(value), 'RED')}) '
                     f'greater than total image count: '
                     f'{R.LR.color_text(str(filecount), 'GRN')}\n'
                     f'Value will be set to '
                     f'{R.LR.color_text(str(filecount), 'GRN')}.\n\n')
            proceed = input(R.LR.color_text('Proceed? (Y | y) -> ', 'ORG'))
            if proceed.strip().casefold() == 'y': value = filecount
            else: continue
        break
    return (True, value)

def display_summary(
        experiment_directory: Path,
        load_epoch: int,
        dataroot: Path,
        test_iterations: int
    ) -> bool:
    R.RENDERER.set_status('Displaying test summary...', 'GRN')
    while True:
        R.RENDERER.reset_console()
        R.LR.println_split(
            f'Experiment directory |', 'GRN',
            f'{experiment_directory.as_posix()}', 'WHT')
        R.LR.println_split(
            f'Load Epoch           |', 'GRN',
            f'{load_epoch}', 'WHT')
        R.LR.println_split(
            f'Dataset Root         |', 'GRN',
            f'{dataroot.as_posix()}', 'WHT')
        R.LR.println_split(
            f'Test Iterations      |', 'GRN',
            f'{test_iterations}', 'WHT')
        
        R.LR.println('\nPlease confirm that these values are correct!', 'ORG')
        R.LR.println_split(
            f'Begin Test  :', 'WHT',
            f'start', 'GRN')
        R.LR.println_split(
            f'Cancel Test :', 'WHT',
            f'exit', 'RED')

        R.LR.println('\nPlease confirm to begin test...')
        selected = input(R.LR.color_text('Begin test? -> ', 'ORG'))
        if selected == 'exit': 
            R.RENDERER.set_status('Exiting...', 'GRN')
            return False
        elif selected == 'start':
            R.RENDERER.set_status('Beginning model test...', 'GRN')
            return True
        else:
            R.RENDERER.set_status(f'Invalid input: {selected}', 'RED')
            continue

def begin_test(
        experiment_directory: Path,
        load_epoch: int,
        dataroot: Path,
        test_iterations: int
    ) -> None:

    # -e --experiment_directory
    # -l --load_epoch
    # -f --config_file
    # -d --dataroot
    # -i --test_iterations
    
    R.LR.println('Test starting. Please wait...', 'ORG')
    proc = subprocess.Popen([
        'python', '-m', 
        'nectargan.start.testing.paired', 
        '-e', experiment_directory.as_posix(),
        '-l', str(load_epoch),
        '-f', CONFIG_PATH.as_posix(),
        '-d', dataroot.as_posix(),
        '-i', str(test_iterations)])
    proc.wait()

def test_finished(experiment_directory: Path) -> None:
    R.RENDERER.set_status(f'Testing complete! Diplaying summary...', 'GRN')
    R.LR.println('Test Completed!', 'GRN')
    R.LR.println_split(
        f'\nResults exported to:', 'GRN',
        f'{Path(experiment_directory, 'test').as_posix()}', 'WHT')
    input(R.LR.color_text('\nPress enter to confirm...', 'ORG'))
    
def run_model_test():
    R.RENDERER.set_status('Configuring model test...', 'GRN')
    success, experiment_directory = get_experiment_directory()
    if not success: return

    success, load_epoch = get_load_epoch(experiment_directory)
    if not success: return

    success, dataroot, test_dir = get_dataroot()
    if not success: return

    success, test_iterations = get_test_iterations(test_dir)
    if not success: return

    confirm = display_summary(
        experiment_directory, load_epoch, dataroot, test_iterations)
    if not confirm: return

    begin_test(experiment_directory, load_epoch, dataroot, test_iterations)
    test_finished(experiment_directory)
    R.RENDERER.reset_status()

