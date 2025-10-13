import importlib

def test_import_top_level() -> None:
    '''Ensure the base package can be imported.'''
    pkg = importlib.import_module('nectargan')
    assert hasattr(pkg, '__version__') or hasattr(pkg, '__file__')

def import_packages(packages: list[str]) -> None:
    for i, mod in enumerate(packages):
        importlib.import_module(mod)
        print(f'Import Successful [{i+1}/{len(packages)}]: {mod}')

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
    print('Testing import of core NectarGAN submodules.')
    import_packages(modules)
    print('Finished importing core submodules.')

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
    print('Testing import of key files.')
    import_packages(key_files)
    print('Finished importing key files.')

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
    print('Testing import of Toolbox submodules.')
    import_packages(modules)
    print('Finished importing Toolbox submodules.')

if __name__ == "__main__":
    test_import_top_level()
    test_import_core_submodules()
    test_import_key_files()
    test_import_toolbox_components()
    print('Import test successful!')
