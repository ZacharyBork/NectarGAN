import json
import pathlib
import random
import numpy as np
import torch
from torchvision.utils import save_image

from .dataset import Pix2pixDataset

def print_losses(epoch: int, iter: int, losses: dict):
    infostring = f'(epoch: {epoch}, iters: {iter})'
    for loss_type in losses:
        infostring += f' {loss_type}: {losses[loss_type]}'
    print(infostring)

def get_config_data(config_filepath: str):
    config_file = pathlib.Path
    if config_filepath == None:
        util_dir = pathlib.Path(__file__).parent
        config_file = pathlib.Path(util_dir, '../config.json').resolve()
    else: config_file = pathlib.Path(config_filepath)
    
    with open(config_file.as_posix(), 'r') as file:
        config_data = json.loads(file.read())['config']
    file.close()
    return config_data

def export_training_config(config: dict, output_directory: pathlib.Path):
    data = { 'config': config }
    existing_configs = [i for i in output_directory.iterdir() if i.suffix == '.json']
    config_export_path = pathlib.Path(output_directory, f'train{str(1 + len(existing_configs))}_config.json')
    with open(config_export_path.as_posix(), 'w') as file:
        json.dump(data, file, indent=4)
    file.close()

def build_experiment_directory(output_directory: str, experiment_name: str, exist_ok: bool):
    output_root = pathlib.Path(output_directory)
    experiment_dir = pathlib.Path(output_root, experiment_name)
    try: experiment_dir.mkdir(parents=False, exist_ok=exist_ok)
    except Exception:
        raise

    examples_dir = pathlib.Path(experiment_dir, 'examples')
    examples_dir.mkdir(exist_ok=exist_ok)

    return experiment_dir, examples_dir

def build_dataloader(config: dict, loader_type: str):
    dataset_path = pathlib.Path(config['COMMON']['DATAROOT'], loader_type).resolve()
    if not dataset_path.exists(): # Make sure data directory exists
        raise Exception(f'Unable to locate dataset at: {dataset_path.as_posix()}')
    dataset = Pix2pixDataset(config=config, root_dir=dataset_path)
    return torch.utils.data.DataLoader(
        dataset, batch_size=config['COMMON']['BATCH_SIZE'], 
        shuffle=True, num_workers=config['COMMON']['NUM_WORKERS'])

def tensor_to_image(tensor):
    '''Converts normalized tensor to uint8 numpy image.'''
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = tensor.detach().cpu().numpy()
    img = np.clip((img + 1) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    return img

def save_examples(config: dict, gen, val_loader: torch.utils.data.DataLoader, epoch: int, output_directory: pathlib.Path):
    gen.eval()

    val_data = list(val_loader.dataset)
    indices = random.sample(range(len(val_data)), config['SAVE']['NUM_EXAMPLES'])
    for i, idx in enumerate(indices):
        x, y = val_data[idx]
        x = x.unsqueeze(0).to(config['COMMON']['DEVICE'])
        y = y.unsqueeze(0).to(config['COMMON']['DEVICE'])

        with torch.no_grad():
            y_fake = gen(x)
        
        save_image(y_fake * 0.5 + 0.5, pathlib.Path(output_directory, f'epoch{epoch}_{str(i+1)}_B_fake.png').as_posix())
        save_image(x * 0.5 + 0.5, pathlib.Path(output_directory, f'epoch{epoch}_{str(i+1)}_A_real.png').as_posix())
        save_image(y * 0.5 + 0.5, pathlib.Path(output_directory, f'epoch{epoch}_{str(i+1)}_B_real.png').as_posix())
    gen.train()

def save_checkpoint(model, optimizer, output_path: pathlib.Path):
    print('=> Saving Checkpoint')
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, output_path.as_posix())

def load_checkpoint(config, gen, opt_gen, disc, opt_disc):
    load_epoch = config['LOAD']['LOAD_EPOCH']
    if load_epoch == -1: base_name = f'final'
    else: base_name = f'epoch{load_epoch}'

    output_directory = pathlib.Path(config['COMMON']['OUTPUT_DIRECTORY'])
    experiment_directory = pathlib.Path(output_directory, config['COMMON']['EXPERIMENT_NAME'])
    
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
        checkpoint = torch.load(path.as_posix(), map_location=config['COMMON']['DEVICE'])
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        for param_group in optimizer.param_groups:
            param_group['lr'] = config['TRAIN']['LEARNING_RATE']

