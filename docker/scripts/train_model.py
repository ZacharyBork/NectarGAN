import time
import subprocess

import renderer as R
import wrapperutils

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

    ln = '---------------------------------------------------------------'
    R.LR.println(f'{ln}\n', 'ORG')
    input(R.LR.color_text('Press enter to confirm...', 'ORG'))

def print_formatted_output(proc, epoch_count) -> None:
    # Takes the output of the training script and makes it look nicer
    config_text = ''
    status_text = ''
    status_update_counter = 0
    status_max_updates = 3
    ln = '---------------------------------------------------------------'
    for line in proc.stdout:
        if line.startswith('(epoch:'):
            R.RENDERER.reset_console()
            R.LR.println(config_text, nocolor=True, line_end='')
            R.LR.println(f'{ln}\n', 'ORG', line_end='')
            for ch in ['(', ')', ':', ',']:
                if ch in line: line = line.replace(ch, '')
            split = line.split(' ')
            current_epoch = split[1]
            R.LR.println_split(
                'Epoch:', 'ORG', f'{current_epoch}/{epoch_count}')
            R.LR.println_split('Iteration:', 'ORG', split[3])
            R.LR.println('Loss:', 'ORG')
            for i in range(round((len(split) - 5) / 2)):
                idx = i * 2 + 5
                R.LR.println_split(
                    f'    - {split[idx]}: ', 'GRN', 
                    f'{split[idx+1]}', 'WHT')
            R.LR.println(f'Progress {ln[:-9]}\n', 'ORG', line_end='')
            progress = int(current_epoch) / epoch_count
            progress_bar = '#' * round(progress * len(ln))
            R.LR.println(progress_bar, 'GRN')
            R.LR.println(f'{ln}\n', 'ORG', line_end='')
            if not status_text == '': 
                R.LR.println(status_text, nocolor=True)
                if status_update_counter == status_max_updates:
                    status_text = ''
                    status_update_counter = 0
                else: status_update_counter += 1
        elif line.startswith('Torch') or line.startswith('CUDA'):
            split = line.split(':')
            newline = f'{R.LR.color_text(split[0], 'ORG')}:'
            newline += f'{R.LR.color_text(split[1], 'GRN')}'
            R.LR.println(newline, nocolor=True, line_end='')
            config_text += newline
        elif line.startswith('LossManager:'):
            split = line.split(' ')
            status_text += f'{R.LR.color_text('LossManager: ', 'ORG')}'
            status_text += f'{R.LR.color_text('Logs updated. ', 'GRN')}'
            status_text += f'{R.LR.color_text('Time taken: ', 'ORG')}'
            status_text += f'{R.LR.color_text(f'{split[-2]} ', 'GRN')}'
            status_text += f'{R.LR.color_text('seconds\n', 'ORG')}'
        elif (line.startswith('Saving example images:') or
              line.startswith('(End of epoch')):
            split = line.split(':')
            status_text += (f'{R.LR.color_text(f'{split[0]}: ', 'ORG')}')
            status_text += f'{R.LR.color_text(f'{split[1]}', 'GRN')}'
        else: R.LR.println(line, line_end='')

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
    
    print_formatted_output(proc, epoch_count)
            
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



