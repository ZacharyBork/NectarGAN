import torch
import pathlib
from .. import config
from torchvision.utils import save_image

def save_examples(gen, val_loader, epoch, output_directory):
    output_dir = pathlib.Path(output_directory)
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5 # Normalize [0:1] from pix2pix [-1:1] output
        save_image(y_fake, pathlib.Path(output_dir, f'y_gen_{epoch}.png').as_posix())
        save_image(x * 0.5 + 0.5, pathlib.Path(output_dir, f'input_{epoch}.png').as_posix())
        if epoch == 1:
            save_image(y * 0.5 + 0.5, pathlib.Path(output_dir, f'label_{epoch}.png').as_posix())
    gen.train()

def save_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    print('=> Saving Checkpoint')
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print('=> Loading Checkpoint')
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr