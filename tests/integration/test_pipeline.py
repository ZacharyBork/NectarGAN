# Tests model training, export, and testing

import json
import subprocess
from pathlib import Path
from typing import Any

from nectargan.utils.rebuild_default_config import DEFINITION

def _get_dataset_directory(root: Path) -> Path:
    dataset_directory = Path(root, 'tests/dataset/noise').resolve()
    assert dataset_directory.exists()
    return dataset_directory

def _build_output_directory(tmp: Path) -> Path:
    output_directory = Path(tmp, 'test_output')
    if not output_directory.exists(): output_directory.mkdir()
    assert output_directory.exists()
    return output_directory

def _update_config_file(
        root: Path,
        dataset_path: Path,
        output_directory: Path
    ) -> tuple[dict[str, Any], Path]:
    print('Updating config file...')

    updated_config = DEFINITION
    cfg = updated_config['config']
    cfg['common']['output_directory'] = output_directory.as_posix()
    cfg['common']['experiment_name'] = 'pipeline_validation_output'
    cfg['dataloader']['dataroot'] = dataset_path.as_posix()
    cfg['train']['generator']['learning_rate']['epochs'] = 2
    cfg['train']['generator']['learning_rate']['epochs_decay'] = 0
    cfg['save']['model_save_rate'] = 1    

    test_config_path = Path(root, 'tests/tmp/config.json').resolve()
    with open(test_config_path, 'w') as file:
        file.write(json.dumps(updated_config, indent=2))
    print('Config file updated.')
    assert test_config_path.exists()
    return test_config_path
    
def _start_training(test_config_path: Path) -> None:
    print('Starting Training...')
    proc = subprocess.Popen(
        [
            'python', '-m', 
            'nectargan.start.training.paired', 
            '-f', test_config_path.as_posix(),
            '-log'
        ],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
        text=True, bufsize=1)
    for line in proc.stdout: print(line, end='')
    return_code = proc.wait()
    assert return_code == 0
    print('Training completed successfully!')

def _start_testing(
        dataset_path: Path,
        test_config_path: Path,
        output_directory: Path
    ) -> None:
    print('Starting testing...')
    experiment_directories = sorted(list(
        output_directory.glob('pipeline_validation_output*')))
    proc = subprocess.Popen(
        [
            'python', '-m', 'nectargan.start.testing.paired', 
            '-e', experiment_directories[-1].as_posix(), 
            '-f', test_config_path.as_posix(),
            '-d', dataset_path.as_posix(),
            '-l', '2',
            '-i', '5'
        ],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
        text=True, bufsize=1)
    for line in proc.stdout: print(line, end='')
    return_code = proc.wait()
    assert return_code == 0
    print('Testing completed successfully!')

def test_pipeline() -> None:
    root = Path(__file__).parent.parent.parent.resolve()
    tmp = Path(root, 'tests/tmp')
    
    dataset_path = _get_dataset_directory(root)
    output_directory = _build_output_directory(tmp)
    test_config_path = _update_config_file(
        root, dataset_path, output_directory)
    _start_training(test_config_path)
    _start_testing(dataset_path, test_config_path, output_directory)

