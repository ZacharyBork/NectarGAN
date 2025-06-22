'''This file serves as a backup definition of the default config.

It also provides a small script to rebuild the default config file in it's 
original location from the `DEFINITION` in this file, just in case it gets lost
or deleted or otherwise broken beyond repair.

Note that, by default, the script in this file will not allow overwriting of an
existing default config if it finds one at `nectargan/config/default.config`.
However, the `-O` (or `--overwrite_existing`) can be used when running the file
to allow it to overwrite the existing default config, effectively restoring it
to its factory default state.
'''

import json
import argparse
from pathlib import Path
from importlib.resources import files

DEFINITION = {
    'config': {
        'common': {
            'device': 'cuda',
            'gpu_ids': [0],
            'output_directory': '',
            'experiment_name': '',
            'experiment_version': 1
        },
        'dataloader': {
            'dataroot': '',
            'direction': 'AtoB',
            'batch_size': 1,
            'num_workers': 0,
            'load' : {
                'input_nc': 3,
                'load_size': 256,
                'crop_size': 256
            },
            'augmentations': {
                'both': {
                    'h_flip_chance': 0.0,
                    'v_flip_chance': 0.0,
                    'rot90_chance': 0.0,
                    'elastic_transform_chance': 0.0,
                    'elastic_transform_alpha': 1.0,
                    'elastic_transform_sigma': 50.0,
                    'optical_distortion_chance': 0.0,
                    'optical_distortion_min': -0.05,
                    'optical_distortion_max': 0.05,
                    'optical_distortion_mode': 'camera',
                    'coarse_dropout_chance': 0.0,
                    'coarse_dropout_holes_min': 1,
                    'coarse_dropout_holes_max': 2,
                    'coarse_dropout_height_min': 0.1,
                    'coarse_dropout_height_max': 0.2,
                    'coarse_dropout_width_min': 0.1,
                    'coarse_dropout_width_max': 0.2
                },
                'input': {
                    'colorjitter_chance': 0.0,
                    'colorjitter_min_brightness': 0.8,
                    'colorjitter_max_brightness': 1.2,
                    'gaussnoise_chance': 0.0,
                    'gaussnoise_min': 0.2,
                    'gaussnoise_max': 0.44,
                    'motionblur_chance': 0.0,
                    'motionblur_limit': 7,
                    'randgamma_chance': 0.0,
                    'randgamma_min': 80.0,
                    'randgamma_max': 120.0,
                    'grayscale_chance': 0.0,
                    'grayscale_method': 'weighted_average',
                    'compression_chance': 0.0,
                    'compression_type': 'jpeg',
                    'compression_quality_min': 99,
                    'compression_quality_max': 100
                },
                'target': {

                }
            }
        },
        'train': {
            'separate_lr_schedules': False,
            'load': {
                'continue_train': False,
                'load_epoch': 1
            },
            'generator': {
                'features': 64,
                'n_downs': 6,
                'block_type': 'UnetBlock',
                'upsample_type': 'Transposed',
                'learning_rate' : {
                    'epochs': 100,
                    'epochs_decay': 100,
                    'initial': 0.0002,
                    'target': 0.0 
                },
                'optimizer': {
                    'beta1': 0.5
                }
            },
            'discriminator': {
                'n_layers': 3,
                'base_channels': 64,
                'max_channels': 512,
                'learning_rate': {
                    'epochs': 100,
                    'epochs_decay': 100,
                    'initial': 0.0002,
                    'target': 0.0
                },
                'optimizer': {
                    'beta1': 0.5
                }    
            },
            'loss': {
                'lambda_gan': 1.0,
                'lambda_l1': 100.0,
                'lambda_l2': 0.0,
                'lambda_sobel': 0.0,
                'lambda_laplacian': 0.0,
                'lambda_vgg': 0.0
            }
        },
        'save': {
            'save_model': True,
            'model_save_rate': 5,
            'auto_increment_version': True,
            'save_examples': True,
            'example_save_rate': 1,
            'num_examples': 1
        },
        'visualizer': {
            'visdom': {
                'enable': False,
                'env_name': 'main',
                'port': 8097,
                'image_size': 400,
                'update_frequency': 40
            }
        }
    }
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-O', '--overwrite_existing', action='store_true',
        help='Allows the script to overwrite an existing default config file.')
    args = parser.parse_args()

    config_path = Path(files('nectargan.config').joinpath('default.json'))
    if config_path.exists():
        if not args.overwrite_existing:
            message = (
                f'Default config file already exists: {config_path}\n\n'
                f'To continue, either delete the existing config file, or use '
                f'"-overwrite_existing" to allow overwriting.')
            raise FileExistsError(message)
    try:
        with open(config_path, 'w') as file:
            json.dump(DEFINITION, file, indent=4)
    except Exception as e:
        raise RuntimeError('Unable to create default config file.') from e