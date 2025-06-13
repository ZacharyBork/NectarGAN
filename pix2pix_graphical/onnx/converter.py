import warnings
from os import PathLike
from pathlib import Path

import torch

from pix2pix_graphical.trainers.trainer import Trainer
from pix2pix_graphical.models.unet.model import UnetGenerator

class ONNXConverter(Trainer):
    def __init__(
            self, 
            experiment_dir: PathLike,
            load_epoch: int,
            in_channels: int=3,
            crop_size: int=256,
            device: str='cpu',
            upsample_block_type: str='Transpose' 
        ) -> None:
        super().__init__(config=None, quicksetup=False)
        self.config.train.load.continue_train = True
        self.config.train.load.load_epoch = load_epoch
        self.load_epoch = load_epoch
        self.experiment_dir = Path(experiment_dir)
        self.in_channels = in_channels
        self.crop = crop_size
        self.device = device
        self.upsample_block_type = upsample_block_type

        self._init_generator()

    def _init_generator(self) -> None:
        '''Initializes generator.'''
        self.gen = UnetGenerator(
            input_size=self.crop, 
            in_channels=self.in_channels,
            upconv_type=self.upsample_block_type)
        
        self.load_checkpoint('G', self.gen)
        self.gen.to(self.device)
        self.gen.eval()

    def _build_output_path(self) -> Path:
        onnx_filename = f'epoch{self.load_epoch}.onnx'
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
        dummy = torch.randn(1, self.in_channels, self.crop, self.crop)
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
        experiment_dir=Path('E:/ML/test_data/pix2pix/output/facades_v1'),
        load_epoch=200,
        in_channels=3,
        crop_size=256,
        device='cpu')
    converter.convert_model()
