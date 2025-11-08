import json
import shutil
from pathlib import Path

import pytest

from nectargan.utils.rebuild_default_config import DEFINITION

ROOT = Path(__file__).parent.parent.resolve()
TMP = Path(ROOT, 'tests/tmp')
CONFIG = Path(TMP, 'config.json')

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--force-cpu", action="store_true", default=False,
        help="Forces config.common.device='cpu' in default NectarGAN config.")

def _cleanup() -> None:
    assert TMP.exists()
    shutil.rmtree(TMP)

def _build_tmp_directory() -> None:
    if TMP.exists(): _cleanup()
    TMP.mkdir()
    assert TMP.exists()

def _build_config() -> Path:
    '''Build a config copy in the tests/tmp directory for testing.'''
    with open(CONFIG, 'w') as file:
        file.write(json.dumps(DEFINITION, indent=2))
    assert CONFIG.exists()

def _force_cpu_device(pytestconfig: pytest.Config) -> None:
    '''Updated NectarGAN default config file to force CPU device.'''
    if pytestconfig.getoption("--force-cpu"):
        with open(CONFIG, 'r') as file: config_data = json.load(file)
        config_data['config']['common']['device'] = 'cpu'
        with open(CONFIG, 'w') as file: 
            file.write(json.dumps(config_data, indent=2))

@pytest.fixture(scope="session", autouse=True)
def session_once(pytestconfig: pytest.Config):
    _build_tmp_directory()
    _build_config()
    initial_device = _force_cpu_device(pytestconfig)
    yield
    _cleanup()

