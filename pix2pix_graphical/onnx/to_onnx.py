# export_onnx.py
import torch
from options.test_options import TestOptions
from models import create_model
import pathlib

if __name__ == '__main__':
    # 1) parse test options (these include --name, --epoch, --dataroot, etc.)
    opt = TestOptions().parse()

    # 2) override a few things to get a clean export
    opt.num_threads = 0       # no data-loading threads
    opt.batch_size = 1        # export batch-size = 1
    opt.serial_batches = True # no shuffle
    opt.no_flip = True        # no image flip
    opt.display_id = -1       # no visdom

    opt.model = 'pix2pix' 
    opt.netD = 'n_layers' 
    opt.n_layers_D = 5 
    opt.netG = 'unet_256' 
    opt.norm = 'instance' 
    opt.direction = 'BtoA' 
    opt.input_nc = 3 
    opt.output_nc = 3 
    opt.num_threads = 4 
    opt.batch_size = 1 
    opt.load_size = 1024 
    opt.crop_size = 1024

    # force CPU for ONNX
    opt.gpu_ids = []

    # 3) create model and load weights from `checkpoints/{opt.name}/`
    model = create_model(opt)
    model.setup(opt)    # load network weights
    model.eval()        # set to inference mode

    # grab the generator network
    netG = model.netG
    netG.to('cpu')

    # 4) build a dummy input: (1 x input_nc x H x W)
    H = getattr(opt, 'crop_size', opt.load_size)
    W = H
    dummy_input = torch.randn(1, opt.input_nc, H, W, dtype=torch.float32)

    # 5) export to ONNX
    output_dir = pathlib.Path(f'./checkpoints/{opt.name}')
    onnx_filename = getattr(opt, 'onnx_file', f"{opt.name}_epoch{opt.epoch}.onnx")
    output_path = pathlib.Path(output_dir, onnx_filename)
    torch.onnx.export(
        netG,
        dummy_input,
        output_path.resolve().as_posix(),
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"ONNX model saved to {onnx_filename}")