# This script can be used to combine the [x, y, y_fake] example images exported
# during training into a single grid image.

from pathlib import Path
from PIL import Image

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--experiment_directory', type=str, 
        help=(
            f'Path to the experiment directory containing the directory of '
            f'example images you would like to combine.'))
    parser.add_argument(
        '-k', '--keep_epochs', type=int, default=10, 
        help=(
            f'Script will only merge example images if the example\'s epoch '
            f'% --keep_epochs == 0. To merge every example, use "-k 1".'))
    parser.add_argument(
        '--stop_on_fail', action='store_true', 
        help=(
            f'If this flag is present and the script encounters an exception '
            f'while loading an example image, it will raise the exception. if '
            f'not, it will skip the iteration and continue.'))
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    experiment_directory = Path(args.experiment_directory)
    if not experiment_directory.exists():
        raise FileNotFoundError(
            f'Unable to located experiment directory at path: '
            f'{experiment_directory.as_posix()}')
    examples_directory = Path(experiment_directory, 'examples')
    if not examples_directory.exists():
        raise FileNotFoundError(
            f'Unable to located examples directory at path: '
            f'{examples_directory.as_posix()}')

    input_images = list(examples_directory.glob('*.png'))
    if len(input_images) == 0:
        raise FileNotFoundError(
            f'No example images found in directory: {examples_directory}')
    # input_images.sort()
    num_epochs = int(input_images[-1].name.split('_')[0].replace('epoch', ''))

    iters = len(list(
        examples_directory.glob(f'epoch{num_epochs}_*_A_real.png')))

    output_dir = Path(experiment_directory, 'examples_combined')
    if not output_dir.exists(): output_dir.mkdir()
    else:
        raise FileExistsError(
            f'Output directory alread exists at: {output_dir.as_posix}\n'
            f'Please delete it to continue.\n\nExiting...')

    for i in range(num_epochs):
        idx = i+1
        if not idx == 1 and not idx%args.keep_epochs == 0: continue
        for j in range(iters):
            iter = j+1
            try:
                input_A_real = Image.open(
                    Path(examples_directory, f'epoch{idx}_{iter}_A_real.png'))
                input_B_fake = Image.open(
                    Path(examples_directory, f'epoch{idx}_{iter}_B_fake.png'))
                input_B_real = Image.open(
                    Path(examples_directory, f'epoch{idx}_{iter}_B_real.png'))
            except Exception: 
                if args.stop_on_fail: raise
                else: continue

            width = input_A_real.width
            output = Path(output_dir, f'epoch{idx}_{iter}.png')
            combined = Image.new('RGB', (width*3, width))
            combined.paste(input_A_real, (0, 0))
            combined.paste(input_B_fake, (width, 0))
            combined.paste(input_B_real, (width * 2, 0))

            combined.save(output)