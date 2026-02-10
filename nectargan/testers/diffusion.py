from os import PathLike
from pathlib import Path
from typing import Literal

from nectargan.trainers import DiffusionTrainer

class DiffusionTester(DiffusionTrainer):
    def __init__(
            self, 
            experiment_directory: PathLike,
            config_file: PathLike | None = None,
            load_step: int | None = None,
            latent_size: int = 64,
            inference_mode: Literal['DDPM', 'DDIM'] = 'DDIM',
            inference_steps: int = 1000,
            sample_ema: bool = True
        ) -> None:
        self._parse_experiment_directory(experiment_directory, config_file)
        self.output_root = self._init_test_output_root()
        super().__init__(self.config_file, log_losses=False, testing=True)

        self.latent_size = latent_size
        self.inference_mode = inference_mode
        self.inference_steps = inference_steps
                
        self.CFG_M.use_ema = sample_ema
        if self.CFG_M.use_ema: self._init_ema()

        if not load_step is None: 
            self.current_step = load_step
            self.config.train.load.load_step = load_step
        self._load_checkpoints()

    def _parse_experiment_directory(
            self, 
            experiment_directory: PathLike,
            config_file: PathLike
        ) -> None:
        experiment_directory = Path(experiment_directory)
        if not experiment_directory.exists():
            raise FileNotFoundError(
                f'Unable to locate experiment directory at path: '
                f'{experiment_directory.as_posix()}')
        self.experiment_dir = experiment_directory
        
        if config_file is None:
            configs = list(experiment_directory.glob('*config.json'))
            if len(configs) == 0:
                raise FileNotFoundError(
                    f'No config files found in experiment directory! '
                    f'Please pass a config_file to the Tester init.')
            self.config_file = configs[-1]
        else: self.config_file = Path(config_file)

    def _init_test_output_root(self) -> Path:
        '''Builds a root output directory, or gets the path to an existing one.
        
        Raises:
            RuntimeError : If unable to create test output root directory.
        '''
        test_dir = Path(self.experiment_dir, 'test')
        if not test_dir.exists(): 
            try: test_dir.mkdir()
            except Exception as e:
                message = 'Unable to create test output directory'
                raise RuntimeError(message) from e
        return test_dir

    def _build_test_output_directory(self) -> None:
        '''Creates an output directory for test results.

        Raises:
            RuntimeError : If unable to create output directory.
        '''
        existing_dirs = list(self.output_root.glob('test_*'))
        if len(existing_dirs) > 0:
            existing_dirs.sort()
            self.test_version = 1 + int(existing_dirs[-1].name.split('_')[-1])
        else: self.test_version = 1
        new_name = f'test_{str(self.test_version).zfill(2)}'
        output_path = Path(self.output_root, new_name)
            
        try: output_path.mkdir()
        except Exception as e:
            raise RuntimeError('Unable to create output directory') from e
        self.output_dir = output_path

    def run_test(
            self,
            image_count: int,
            batches: int = 1,
            caption: str | None = None,
            cfg_scale: float = 7.5
        ) -> None:
        self._build_test_output_directory()
        if not caption is None: context, _ = self.model.text_encoder([caption])
        else: context = None

        was_training = self.model.training
        self.model.eval()
        try: 
            for i in range(image_count):
                print(f'Running iterations: {i+1} / {image_count}')
                self._save_inference_example(
                    idx=i, output_directory=self.output_dir, batches=batches, 
                    latent_spatial_size=self.latent_size, context=context, 
                    cfg_scale=cfg_scale, mode=self.inference_mode,
                    inference_steps=self.inference_steps)
        finally: self.model.train(was_training)

    

