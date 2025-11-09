import time
import subprocess

import renderer as R
import wrapperutils
import formatter

def _get_loss_subspec() -> tuple[bool, str]:
    loss_subspecs = ['basic', 'basic+vgg', 'extended', 'extended+vgg']
    while True:
        R.RENDERER.reset_console()
        msg = (
            f'Please enter a loss subspec to use for training, or "exit" to '
            f'quit. Valid subspecs are:\n')
        R.LR.println(msg)
        for i in loss_subspecs: R.LR.println(i, 'GRN')

        R.LR.println(
            f'\nSee here for more info:\nhttps://github.com/ZacharyBork/'
            f'NectarGAN/blob/main/docs/api/losses/loss_spec.md')
        R.LR.println('\nPlease enter valid loss subspec...')
        selected = input(R.LR.color_text('Loss Subspec -> ', 'ORG'))
        selected = selected.strip().casefold()
        if selected == 'exit': return (False, None)
        if not selected in loss_subspecs:
            R.RENDERER.set_status(f'Subspec not valid: {selected}', 'RED')
            continue
        break
    return (True, selected)

def _get_should_log_losses() -> tuple[bool, bool]:
    while True:
        R.RENDERER.reset_console()
        msg = (
            f'Should a loss log be generated for this training session? '
            f'Enter ("y" | "yes") or ("n" | "no"), or "exit" to quit.')
        R.LR.println(msg)

        R.LR.println('\nPlease enter a response...')
        selected = input(R.LR.color_text('Write Loss Logs? -> ', 'ORG'))
        selected = selected.strip().casefold()
        match selected:
            case 'exit': return (False, None)
            case 'y' | 'yes': output = True
            case 'n' | 'no': output = False
            case _:
                R.RENDERER.set_status(f'Answer not valid: {selected}', 'RED')
                continue
        break
    return (True, output)

def _display_summary(
        subspec: str,
        log_losses: bool
    ) -> bool:
    R.RENDERER.set_status('Displaying training summary...', 'GRN')
    while True:
        R.RENDERER.reset_console()
        R.LR.println_split(
            f'Config File Path |', 'GRN',
            f'{wrapperutils.get_config_path().as_posix()}', 'WHT')
        R.LR.println_split(
            f'Loss Subspec     |', 'GRN',
            f'{subspec}', 'WHT')
        R.LR.println_split(
            f'Log Losses       |', 'GRN',
            f'{str(log_losses)}', 'WHT')
        
        R.LR.println('\nPlease confirm that these values are correct!', 'ORG')
        R.LR.println_split(
            f'Begin Training  :', 'WHT',
            f'start', 'GRN')
        R.LR.println_split(
            f'Cancel Training :', 'WHT',
            f'exit', 'RED')

        R.LR.println('\nPlease confirm to begin training...')
        selected = input(R.LR.color_text('Begin training? -> ', 'ORG'))
        if selected == 'exit': 
            R.RENDERER.set_status('Exiting...', 'GRN')
            return False
        elif selected == 'start':
            R.RENDERER.set_status('Beginning training session...', 'GRN')
            return True
        else:
            R.RENDERER.set_status(f'Invalid input: {selected}', 'RED')
            continue

def display_training_finished(train_length: float) -> None:
    R.RENDERER.set_status('Training completed sucessfully!', 'GRN')
    R.LR.println_split(
        f'Time Taken |', 'GRN',
        f'{time.strftime('%H:%M:%S', time.gmtime(train_length))}', 'WHT')

    R.RENDERER.add_divider()
    input(R.LR.color_text('Press enter to confirm...', 'ORG'))

def begin_training() -> None:
    success, subspec = _get_loss_subspec()
    if not success: return
    
    success, log_losses = _get_should_log_losses()
    if not success: return

    confirm = _display_summary(subspec, log_losses)
    if not confirm: return

    cmd = [
        'python', '-u', '-m', 
        'nectargan.start.training.paired', 
        '-f', wrapperutils.get_config_path().as_posix(),
        '-lss', subspec]
    if log_losses: cmd.append('-log')

    R.RENDERER.set_status('Training...', 'GRN')
    gen_epochs = int(wrapperutils.get_config_value(
        ['config', 'train', 'generator', 'learning_rate', 'epochs']))
    gen_epochs_decay = int(wrapperutils.get_config_value(
        ['config', 'train', 'generator', 'learning_rate', 'epochs']))
    epoch_count = gen_epochs + gen_epochs_decay
    start_time = time.time()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
        text=True, bufsize=1)
    
    formatter.train_log(proc, epoch_count)
            
    return_code = proc.wait()
    if not return_code == 0:
        R.RENDERER.set_status(f'Training canceled...', 'RED')
        R.LR.println(
            f'Unable to start training. Please confirm that your dataset '
            f'directory path is set correctly in your config file, or use the '
            f'"dataset-set" command to the current dataset.\n')
        input(R.LR.color_text('Press enter to confirm...', 'ORG'))
        return
    train_length = time.time() - start_time
    display_training_finished(train_length)



