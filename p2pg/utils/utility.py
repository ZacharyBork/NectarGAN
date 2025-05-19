import torch
import json
import pathlib
from .dataset import Pix2pixDataset
from torchvision.utils import save_image

def get_config_data():
    util_dir = pathlib.Path(__file__).parent
    config_file = pathlib.Path(util_dir, '../config.json').resolve()
    
    with open(config_file.as_posix(), 'r') as file:
        config_data = json.loads(file.read())['config']
    file.close()
    return config_data

def export_training_config(config: dict, output_directory: pathlib.Path):
    data = { 'config': config }
    config_export_path = pathlib.Path(output_directory, f'train{str(1)}_config.json')
    with open(config_export_path.as_posix(), 'w') as file:
        json.dump(data, file, indent=4)
    file.close()

def build_experiment_directory(output_directory: str, experiment_name: str):
    output_root = pathlib.Path(output_directory)
    experiment_dir = pathlib.Path(output_root, experiment_name)
    try: experiment_dir.mkdir(parents=False, exist_ok=False)
    except Exception:
        raise

    examples_dir = pathlib.Path(experiment_dir, 'examples')
    examples_dir.mkdir()

    return experiment_dir, examples_dir

def build_dataloader(config: dict, loader_type: str):
    dataset_path = pathlib.Path(config['DATAROOT'], loader_type).resolve()
    if not dataset_path.exists(): # Make sure data directory exists
        raise Exception(f'Unable to locate dataset at: {dataset_path.as_posix()}')
    dataset = Pix2pixDataset(config=config, root_dir=dataset_path)
    return torch.utils.data.DataLoader(dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=config['NUM_WORKERS'])

def save_examples(config: dict, gen, val_loader: torch.utils.data.DataLoader, epoch: int, output_directory: pathlib.Path):
    x, y = next(iter(val_loader))
    x, y = x.to(config['DEVICE']), y.to(config['DEVICE'])
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        save_image(y_fake * 0.5 + 0.5, pathlib.Path(output_directory, f'epoch{epoch}_fake_B.png').as_posix())
        save_image(x * 0.5 + 0.5, pathlib.Path(output_directory, f'epoch{epoch}_real_A.png').as_posix())
        save_image(y * 0.5 + 0.5, pathlib.Path(output_directory, f'epoch{epoch}_real_B.png').as_posix())
    gen.train()

def save_checkpoint(model, optimizer, output_path: pathlib.Path):
    print('=> Saving Checkpoint')
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, output_path.as_posix())

def load_checkpoint(config: dict, checkpoint_file, model, optimizer, lr):
    print('=> Loading Checkpoint')
    checkpoint = torch.load(checkpoint_file, map_location=config['DEVICE'])
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# def load_checkpoint(checkpoint_file, model, optimizer, lr):
#     print('=> Loading Checkpoint')
#     checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])

#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
