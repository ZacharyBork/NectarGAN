import json
from pathlib import Path
from os import PathLike
from typing import Type, TypeVar, Any
from dataclasses import is_dataclass
from importlib.resources import files

from nectargan.config import Config, GANConfig, DiffusionConfig
T = TypeVar('T', bound=Config)

class ConfigManager():
    def __init__(
            self, 
            input_config: str | PathLike | dict[str, Any] | None=None
        ) -> None:
        self.config_type: str = None

        if isinstance(input_config, dict):
            self._validate_config_dict(input_config)
            self.raw = input_config
        else: self.raw = self.parse_config_file(input_config)
        self.data = self.load_full_config()

    def _match_keys(
            self, 
            jsonA: dict[str, Any], 
            jsonB: dict[str, Any]
        ) -> None:
        '''Performs a simple structure check to validate a config dict.'''
        def _get_all_keys(data, _keys=None):
            if _keys is None: keys = []
            else: keys = _keys
            for key in data:
                keys.append(key)
                if isinstance(data[key], dict):
                    _get_all_keys(data[key], _keys=keys)
            return keys
        keysA = _get_all_keys(jsonA)
        keysB = _get_all_keys(jsonB)
        if not keysA == keysB:
            print(keysB)
            raise ValueError('Input config dict is invalid.')
        
    def _get_default_config_file(self, relative_path: str) -> Path:
        trav = files('nectargan.config').joinpath(relative_path)
        full_path = Path(trav).resolve()
        if not full_path.exists():
            raise FileNotFoundError('' \
                f'Unable to locate default config file at path: '
                f'{full_path.as_posix}')
        return full_path

    def _try_get_config_type(self, input_data: dict[str, Any]) -> None:
        try: cfg_type = input_data['config_type']
        except KeyError:
            raise KeyError(f'Unable to locate config_type key in config dict.')
        
        match cfg_type:
            case 'pix2pix': C = self.config_type = GANConfig
            case 'diffusion': C = self.config_type = DiffusionConfig
            case _:
                raise KeyError(f'Invalid config_type key: {cfg_type}')
        self.config_type_name = cfg_type
        self.default_file_path = self._get_default_config_file(C.DEFAULT_FILE)
        self.group_schema = C.GROUP_SCHEMA

    def _validate_config_dict(self, input_data: dict[str, Any]) -> None:
        assert isinstance(input_data, dict)
        self._try_get_config_type(input_data)
        self._get_default_config_file()
        
        with open(self.default_file_path.as_posix(), 'r') as file:
            default_data = json.loads(file.read())['config']
        return self._match_keys(jsonA=default_data, jsonB=input_data)

    def parse_config_file(
            self, 
            config_filepath: str | PathLike | None
        ) -> dict:
        '''Loads a JSON config file and returns the raw config data.
        
        config_filepath can also be None. In this case, this function will 
        instead try to load the default config file from "./default.json". If 
        it is unable to do that, it will raise a FileNotFoundError.

        Args:
            config_filepath : Absolute file path of config JSON, or None.

        Returns:
            dict : The raw JSON data imported from the config file.

        Raises:
            FileNotFoundError : If provided config file does not exist and/ or 
                default config is missing.
        '''
        config_file = Path
        if config_filepath == None:
            self.config_type = 'pix2pix'
            config_file = self._get_default_config_file('default.json')
        else: 
            config_file = Path(config_filepath)
            if not config_file.exists():
                raise FileNotFoundError(
                    f'Unable to locate config file at: '
                    f'{config_file.as_posix()}')

        with open(config_file.as_posix(), 'r') as file:
            config_data = json.loads(file.read())['config']
        self._try_get_config_type(config_data)
        return config_data
    
    def load_config_section(self, data: dict[str, Any], cls: Type[T]) -> T:
        '''Loops through dataclass and assigns corresponding value from dict.
        
        Args:
            data : Dict of data from a section of the config file (e.g. COMMON, 
                TRAIN, etc.).
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
    
    def load_full_config(self) -> Config:
        '''Initializes a config_data.Config dataclass from config JSON data.
        
        Returns:
            config_data.Config : Config dataclass with all nested 
                sub-dataclasses.
        '''
        sections = {}
        for name, cls in self.group_schema.items():
            current_section = self.raw[name]
            sections[name] = self.load_config_section(current_section, cls)

        return self.config_type(config_type=self.config_type_name, **sections)
    
    def export_config(self, output_directory: Path):
        '''Exports a copy of the config to the provided output directory.
        
        This function will first check the files in the provided output 
        directory looking for previously exported config files. If it finds 
        any, it will increment the name accordingly. The files are named 
        trainX_config.json to help automatically keep track of settings during 
        fine tuning of models.

        Args:
            output_directory : Directory to export config file to.

        Raises:
            FileNotFoundError : If output directory does not exist.
        '''
        if not output_directory.exists():
            msg = 'Unable to export config file to: {}'
            raise FileNotFoundError(msg.format(output_directory.as_posix()))
        data = { 'config': self.raw }
        existing_configs = [i for i in output_directory.glob('*config.json')]
        filename = f'train{str(1 + len(existing_configs))}_config.json'
        config_path = Path(output_directory, filename)
        try:
            with open(config_path.as_posix(), 'w') as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            message = 'Unable to export config file: {}'
            raise Exception(message.format(config_path.as_posix())) from e

    
if __name__ == "__main__":
    root = Path(__file__).parent.resolve()
    for x in ['default.json', 'defaults/diffusion.json']:
        config = Path(root, x)
        config_manager = ConfigManager(input_config=config.as_posix())
        print(f'Config: {x}\n\n' + str(config_manager.data) + '\n')


