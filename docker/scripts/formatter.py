import renderer as R

def _format_train_system_line(line: str) -> str:
    split = line.split(':')
    newline = f'{R.LR.color_text(split[0], 'ORG')}:'
    newline += f'{R.LR.color_text(split[1], 'GRN')}'
    R.LR.println(newline, nocolor=True, line_end='')
    return newline

def _format_train_lossmanager_line(line: str) -> str:
    split = line.split(' ')
    newline = f'{R.LR.color_text('LossManager: ', 'ORG')}'
    newline += f'{R.LR.color_text('Logs updated. ', 'GRN')}'
    newline += f'{R.LR.color_text('Time taken: ', 'ORG')}'
    newline += f'{R.LR.color_text(f'{split[-2]} ', 'GRN')}'
    newline += f'{R.LR.color_text('seconds\n', 'ORG')}'
    return newline

def _format_train_saving_examples_line(line: str) -> str:
    split = line.split(':')
    newline = (f'{R.LR.color_text(f'{split[0]}: ', 'ORG')}')
    newline += f'{R.LR.color_text(f'{split[1]}', 'GRN')}'
    return newline

def _update_training_interface_core(
        line: str,
        config_text: str, 
        epoch_count: int,
    ) -> None:
    R.RENDERER.reset_console()
    R.LR.println(config_text, nocolor=True, line_end='')
    R.RENDERER.add_divider()
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
    R.RENDERER.render_progress_bar_full(
        label='Progress',
        current_value=int(current_epoch), 
        total_value=epoch_count)
    
def train_log(proc, epoch_count) -> None:
    # Takes the output of the training script and makes it look nicer
    config_text = ''
    status_text = ''
    status_update_counter = 0
    status_max_updates = 3
    for line in proc.stdout:
        if line.startswith('(epoch:'):
            _update_training_interface_core(line, config_text, epoch_count)
            if not status_text == '': 
                R.LR.println(status_text, nocolor=True)
                if status_update_counter == status_max_updates:
                    status_text = ''
                    status_update_counter = 0
                else: status_update_counter += 1
        elif line.startswith('Torch') or line.startswith('CUDA'):
            config_text += _format_train_system_line(line)
        elif line.startswith('LossManager:'):
            status_text += _format_train_lossmanager_line(line)
        elif (line.startswith('Saving example images:') or
              line.startswith('(End of epoch')):
            status_text += _format_train_saving_examples_line(line)
        else: R.LR.println(line, line_end='')