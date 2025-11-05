import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
CONFIG = Path(ROOT, 'nectargan/config/default.json')

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--force-cpu", action="store_true", default=False,
        help="Forces config.common.device='cpu' in default NectarGAN config.")

def _force_cpu_device(pytestconfig: pytest.Config) -> str:
    '''Updated NectarGAN default config file to force CPU device.'''
    if pytestconfig.getoption("--force-cpu"):
        with open(CONFIG, 'r') as file: config_data = json.load(file)
        initial_device = config_data['config']['common']['device']
        config_data['config']['common']['device'] = 'cpu'

        with open(CONFIG, 'w') as file: 
            file.write(json.dumps(config_data, indent=4))
        return initial_device

def _reset_device(pytestconfig: pytest.Config, initial_device: str) -> str:
    '''Reverts device in default config file to its pre-testing state.'''
    if pytestconfig.getoption("--force-cpu"):
        if not initial_device is None and not initial_device == 'cpu':
            with open(CONFIG, 'r') as file: config_data = json.load(file)
            config_data['config']['common']['device'] = initial_device
            with open(CONFIG, 'w') as file: 
                file.write(json.dumps(config_data, indent=4))

@pytest.fixture(scope="session", autouse=True)
def session_once(pytestconfig: pytest.Config):
    initial_device = _force_cpu_device(pytestconfig)
    yield
    _reset_device(pytestconfig, initial_device)

