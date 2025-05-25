import pathlib
from typing import Union

from pix2pix_graphical.config.config_manager import ConfigManager
from pix2pix_graphical.visualizer.visdom_visualizer import VisdomVisualizer

class Trainer():
    def __init__(self, config_filepath: Union[str, None]=None) -> None:
        self.config_manager = ConfigManager(config_filepath)  # Init config manager
        self.config = self.config_manager.data # Store config data for easier access
        self.init_visualizers()   # Init visualizer

    def init_visualizers(self) -> None:
        vcon = self.config.visualizer # Get visualizer config data
        if vcon.enable_visdom:        # Init Visdom visualizer
            self.vis = VisdomVisualizer(env=vcon.visdom_env_name) 
            self.vis.clear_env()      # Clear Visdom environment    

    def build_output_directory(self) -> None:
        '''Builds an output directory structure for the experiment.'''
        output_root = pathlib.Path(self.config.common.output_directory)
        exp_name = self.config.common.experiment_name       # Get experiment name
        exp_version = self.config.common.experiment_version # Get experiment version
        name = f'{exp_name}_v{exp_version}'                 # Build output dir name
        experiment_dir = pathlib.Path(output_root, name)    # Output dir path
        
        try: # Exists okay is true if we're loading a previously trained model
            experiment_dir.mkdir(parents=False, exist_ok=self.config.load.continue_train)
        except FileExistsError as e: # Otherwise we check to see if auto-increment is on
            if self.config.save.auto_increment_version:
                try:
                    prev_versions = [i.name for i in experiment_dir.parent.iterdir() if exp_name in i.name]
                    prev_versions.sort()
                    new_version = 1 + int(prev_versions[-1].split('v')[-1])
                    experiment_dir = pathlib.Path(output_root, f'{exp_name}_v{new_version}')
                    
                    try: experiment_dir.mkdir(parents=False, exist_ok=False)
                    except Exception: raise
                except Exception as x: raise RuntimeError('Unable to overwrite experiment dir') from x
            else:
                error_message = (
                    f'Unable to save experiment: '
                    f'Experiment directory already exists. '
                    f'Overwriting can be enabled via config. '
                    f'This will delete existing experiment data, though!')
                raise RuntimeError(error_message) from e

        examples_dir = pathlib.Path(experiment_dir, 'examples')
        examples_dir.mkdir(exist_ok=self.config.load.continue_train)
        self.experiment_dir, self.examples_dir =  experiment_dir, examples_dir

    def export_config(self) -> None:
        '''Exports a versioned config JSON file to the experiment output directory.
        
        This function just abstracts ConfigManager.export_config for ease of use with
        instances of trainer classes. This is also not strictly necessary to do. The
        Trainer's config data is intialized from the input config file. This just makes
        a copy of it in the experiment output directory for notekeeping purposes.
        '''
        self.config_manager.export_config(self.experiment_dir)

    def print_end_of_epoch(self, index: int, begin_epoch: float, end_epoch: float) -> None:
        print(f'(End of epoch {index}) Time: {end_epoch - begin_epoch:.2f} seconds', flush=True)
