import warnings
from os import PathLike
from pathlib import Path
from typing import Any

import torch

from nectargan.trainers.trainer import Trainer
from nectargan.models.unet.model import UnetGenerator
import nectargan.models.unet.blocks as unetblocks 
from nectargan.config.config_manager import ConfigManager

class ONNXConverter(Trainer):
    def __init__(
            self, 
            experiment_dir: PathLike,
            config: str | PathLike | ConfigManager | dict[str, Any],
            load_epoch: int | None=None,
            in_channels: int | None=None,
            crop_size: int | None=None,
            device: str | None=None,
        ) -> None:
        if config is None:     # Trainer will happily take a NoneType config
            raise ValueError(( # but ONNXConverter can't so we stop that here
                f'ONNXConverter config cannot be NoneType. '
                f'Please input a valid config to continue.'))
        super().__init__(config=config, quicksetup=False)
        self.config.train.load.continue_train = True # Force continue_train
        self.experiment_dir = Path(experiment_dir)
        if not load_epoch is None: # Override load_epoch if not NoneType
            self.config.train.load.load_epoch = load_epoch
        self.crop_size = crop_size or self.config.dataloader.load.crop_size
        self.device = device or self.device
        self.in_channels = in_channels or self.config.dataloader.load.input_nc
        
        self._init_generator()

    def _init_generator(self) -> None:
        '''Initializes generator.'''
        match self.config.train.generator.block_type:
            case 'UnetBlock': block_type = unetblocks.UnetBlock
            case 'ResidualUnetBlock': block_type = unetblocks.ResidualUnetBlock
            case _: block_type = unetblocks.UnetBlock

        self.gen = UnetGenerator(
            input_size=self.crop_size, 
            in_channels=self.in_channels,
            features=self.config.train.generator.features,
            n_downs=self.config.train.generator.n_downs,
            block_type=block_type,
            upconv_type=self.config.train.generator.upsample_type)
        
        self.load_checkpoint('G', self.gen)
        self.gen.to(self.device)
        self.gen.eval()

    def _build_output_path(self) -> Path:
        onnx_filename = f'epoch{self.config.train.load.load_epoch}.onnx'
        return Path(self.experiment_dir, onnx_filename)

    def convert_model(
            self,
            export_params: bool=True,
            opset_version: int=17,
            do_constant_folding: bool=True,
            input_names: list[str]=['input'],
            output_names: list[str]=['output'],
            suppress_onnx_warnings: bool=True
        ) -> str:
        dummy = torch.randn(
            1, self.in_channels, self.crop_size, self.crop_size)
        dummy.to(self.device)

        output_path = self._build_output_path()

        if suppress_onnx_warnings:
            # See: https://github.com/pytorch/pytorch/issues/75252
            # Silences the ONNX export training mode warning.
            warnings.filterwarnings('ignore', message='.*ONNX export mode*')
            if opset_version > 10:
                warnings.filterwarnings(
                    'ignore', message='.*onnx::Slice op. Constant folding*')
        torch.onnx.export(
            self.gen, dummy, output_path.resolve().as_posix(),
            export_params=export_params, 
            training=torch.onnx.TrainingMode.EVAL,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            input_names=input_names, output_names=output_names)
        return output_path.as_posix()

if __name__ == "__main__":
    converter = ONNXConverter(
        experiment_dir=Path('E:/ML/test_data/pix2pix/output/facades_v3'),
        config= Path(
            'E:/ML/test_data/pix2pix/output/facades_v3/train1_config.json'),
        load_epoch=100,
        device='cpu')
    converter.convert_model()
