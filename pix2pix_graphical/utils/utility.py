import pathlib
import numpy as np
import torch

from pix2pix_graphical.config.config_data import Config
from pix2pix_graphical.dataset.pix2pix_dataset import Pix2pixDataset

def print_losses(epoch: int, iter: int, losses: dict):
    infostring = f'(epoch: {epoch}, iters: {iter})'
    for loss_type in losses:
        infostring += f' {loss_type}: {losses[loss_type]}'
    print(infostring)
    

def build_dataloader(config: Config, loader_type: str):
    '''Initializes a Torch dataloader of the given type from a Pix2pixDataset.'''
    dataset_path = pathlib.Path(config.common.dataroot, loader_type).resolve()
    if not dataset_path.exists(): # Make sure data directory exists
        raise Exception(f'Unable to locate dataset at: {dataset_path.as_posix()}')
    dataset = Pix2pixDataset(config=config, root_dir=dataset_path)
    return torch.utils.data.DataLoader(
        dataset, batch_size=config.dataloader.batch_size, 
        shuffle=True, num_workers=config.dataloader.num_workers)

def tensor_to_image(tensor):
    '''Converts normalized tensor to uint8 numpy image.'''
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = tensor.detach().cpu().numpy()
    img = np.clip((img + 1) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    return img    

def save_checkpoint(model, optimizer, output_path: pathlib.Path):
    print('=> Saving Checkpoint')
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, output_path.as_posix())

def load_checkpoint(config: Config, gen, opt_gen, disc, opt_disc):
    load_epoch = config.load.load_epoch
    if load_epoch == -1: base_name = f'final'
    else: base_name = f'epoch{load_epoch}'

    output_directory = pathlib.Path(config.common.output_directory)
    experiment_directory = pathlib.Path(output_directory, config.common.experiment_name)
    
    # Load generator checkpoint
    checkpoint_gen_path = pathlib.Path(experiment_directory, f'{base_name}_netG.pth.tar')
    if not checkpoint_gen_path.exists():
        raise Exception(f'Unable to locate generator checkpoint at: {checkpoint_gen_path.as_posix()}')
    
    # Load discriminator checkpoint
    checkpoint_disc_path = pathlib.Path(experiment_directory, f'{base_name}_netD.pth.tar')
    if not checkpoint_disc_path.exists():
        raise Exception(f'Unable to locate discriminator checkpoint at: {checkpoint_disc_path.as_posix()}')
    
    models = [(checkpoint_gen_path, gen, opt_gen), (checkpoint_disc_path, disc, opt_disc)]
    for path, model, optimizer in models:
        checkpoint = torch.load(path.as_posix(), map_location=config.common.device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        for param_group in optimizer.param_groups:
            param_group['lr'] = config.train.learning_rate

