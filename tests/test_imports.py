import importlib

def test_import_top_level() -> None:
    '''Ensure the base package can be imported.'''
    pkg = importlib.import_module('nectargan')
    assert hasattr(pkg, '__version__') or hasattr(pkg, '__file__')

def _import_packages(packages: list[str]) -> None:
    '''Test imports a list of packages by their path.
    
    Args:
        packages : List of submodule paths to import.
    '''
    for i, mod in enumerate(packages):
        importlib.import_module(mod)

def test_import_core_submodules():
    '''Ensure all main submodules import cleanly.'''
    modules = [
        'nectargan.config',
        'nectargan.dataset',
        'nectargan.losses',
        'nectargan.models',
        'nectargan.models.patchgan',
        'nectargan.models.unet',
        'nectargan.onnx',
        'nectargan.scheduling',
        'nectargan.testers',
        'nectargan.trainers',
        'nectargan.utils',
        'nectargan.visualizer',
        'nectargan.visualizer.visdom',
    ]
    _import_packages(modules)

def test_import_key_files() -> None:
    '''Import individual key components directly.'''
    key_files = [
        'nectargan.config.config_manager',
        'nectargan.dataset.paired_dataset',
        'nectargan.losses.loss_manager',
        'nectargan.losses.lm_data',
        'nectargan.losses.losses',
        'nectargan.losses.pix2pix_objective',
        'nectargan.models.patchgan.model',
        'nectargan.models.patchgan.blocks',
        'nectargan.models.unet.model',
        'nectargan.models.unet.blocks',
        'nectargan.scheduling.data',
        'nectargan.scheduling.scheduler',
        'nectargan.scheduling.scheduler_torch',
        'nectargan.scheduling.schedules',
        'nectargan.testers.tester',
        'nectargan.trainers.trainer',
        'nectargan.trainers.pix2pix_trainer',
    ]
    _import_packages(key_files)

def test_import_toolbox_components() -> None:
    '''Ensure all toolbox submodules import cleanly.'''
    modules = [
        'nectargan.toolbox',
        'nectargan.toolbox.components',
        'nectargan.toolbox.helpers',
        'nectargan.toolbox.utils',
        'nectargan.toolbox.widgets',
        'nectargan.toolbox.workers',
    ]
    _import_packages(modules)

