import pathlib
import json
from typing import Type, TypeVar, Union, Any
from dataclasses import is_dataclass

from . import config_data
T = TypeVar('T')
GROUP_SCHEMA = {
    'common': config_data.ConfigCommon,
    'dataloader': config_data.ConfigDataloader,
    'train': config_data.ConfigTrain,
    'loss': config_data.ConfigLoss,
    'generator': config_data.ConfigGenerator,
    'discriminator': config_data.ConfigDiscriminator,
    'optimizer': config_data.ConfigOptimizer,
    'load': config_data.ConfigLoad,
    'save': config_data.ConfigSave,
    'visualizer': config_data.ConfigVisualizer,
}

class ConfigManager():
    def __init__(self, config_filepath: Union[str, None]=None) -> None:
        self.raw = self.parse_config_file(config_filepath)
        self.data = self.load_full_config()

    def parse_config_file(self, config_filepath: Union[str, None]=None) -> dict:
        '''Loads a JSON config file and returns the raw config data.
        
        config_filepath can also be None. In this case, this function will instead try
        to load the default config file located at "/pix2pix-graphical/config.json". If
        it is unable to do that, it will raise a FileNotFoundError.

        Args:
            config_filepath : Absolute file path of config JSON, or None.

        Returns:
            dict : The raw JSON data imported from the config file as a dict[str, Any].

        Raises:
            FileNotFoundError : If provided config file does not exist and/ or 
                default config is missing.
        '''
        config_file = pathlib.Path
        if config_filepath == None:
            script_root = pathlib.Path(__file__).parent
            config_file = pathlib.Path(script_root, '../../config.json').resolve()
        else: config_file = pathlib.Path(config_filepath)
        
        if not config_file.exists():
            raise FileNotFoundError(f'Unable to locate config file at: {config_file.as_posix()}')

        with open(config_file.as_posix(), 'r') as file:
            config_data = json.loads(file.read())['config']
        file.close()
        return config_data
    
    def load_config_section(self, data: dict[str, Any], cls: Type[T]) -> T:
        '''Loops through the provided dataclass and assigns the corresponding value from the data dict.
        
        Args:
            data : Dict of data from a section of the config file (e.g. COMMON, TRAIN, etc.).
            cls : Dataclass to initialize values for.

        Returns:
            T : Instance of the input dataclass type.

        Raises:
            TypeError : If provided class is not a dataclass
        '''
        if not is_dataclass(cls):
            raise TypeError(f'{cls} is not a dataclass')
        
        field_values = {}
        for field in cls.__dataclass_fields__:
            field_type = cls.__dataclass_fields__[field].type
            value = data.get(field)

            # Allow for recursive loading of more deeply nested dataclasses
            if is_dataclass(field_type) and isinstance(value, dict):
                value = self.load_config_section(value, field_type)

            field_values[field] = value

        return cls(**field_values)
    
    def load_full_config(self) -> config_data.Config:
        '''Initializes a config_data.Config dataclass from config JSON data.
        
        Returns:
            config_data.Config : Config dataclass with all nested sub-dataclasses.
        '''
        sections = {}
        for name, cls in GROUP_SCHEMA.items():
            current_section = self.raw[name]
            sections[name] = self.load_config_section(current_section, cls)

        return config_data.Config(**sections)
    
    def export_config(self, output_directory: pathlib.Path):
        '''Exports a copy of the current config file to the provided output directory.
        
        This function will first check the files in the provided output directory looking
        for previously exported config files. If it finds any, it will increment the name
        accordingly. The files are named trainX_config.json to help automatically keep 
        track of settings during fine tuning of models.

        Args:
            output_directory : Directory to export config file to.

        Raises:
            FileNotFoundError : If output directory does not exist.
        '''
        if not output_directory.exists():
            raise FileNotFoundError(f'Unable to export config file to: {output_directory.as_posix()}')
        data = { 'config': self.raw }
        existing_configs = [i for i in output_directory.iterdir() if i.suffix == '.json']
        config_export_path = pathlib.Path(output_directory, f'train{str(1 + len(existing_configs))}_config.json')
        try:
            with open(config_export_path.as_posix(), 'w') as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            raise Exception(
                f'Unable to export config file: {config_export_path.as_posix()}') from e

    
if __name__ == '__main__':
    config = ConfigManager()
    print(config.data)
