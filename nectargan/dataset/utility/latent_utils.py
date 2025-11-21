import sys
from os import PathLike
from pathlib import Path

from nectargan.config import DiffusionConfig

def validate_latent_size(
        size: int,
        input_size: int,
        divisor: int
    ) -> None:
    if size < 2:
        raise ValueError(
            f'Latent size divisor ({divisor}) too large for input size '
            f'({input_size}). The resulting latent tensor would have '
            f'(W=1, H=1).\n\nPlease increase input size, or decrease '
            f'`latent_size_divisor`.')
    elif size < 4:
        print(
            f'Warning: The current input size and latent size divisor '
            f'will result in a very small spacial size for the latent '
            f'space tensor at the DAE\'s bottleneck layer.\n\nThis can '
            f'lead to very poor training results! Please consider '
            f'increasing input size, or decreasing `latent_size_divisor`.')

def get_latent_spatial_size(config: DiffusionConfig) -> int:
    '''Derive latent spatial size from input size and divisor.'''
    cfg = config.model
    match cfg.model_type:
        case 'latent': cfg = cfg.latent
        case 'stable': cfg = cfg.stable
    if not cfg.override_latent_size:
        size = round(cfg.input_size / max(1, cfg.latent_size_divisor))
        validate_latent_size(
            size, cfg.input_size, cfg.latent_size_divisor)
    else: size = cfg.latent_size
    return size

def init_latent_cache(
        dataroot: PathLike, 
        cache_name: str='train'
    ) -> tuple[Path, bool]:
    '''Inits/validates latent cache output directory.
    
    Args:
        dataroot : A PathLike object pointing to the root directory of the
            dataset to cache.
    
    Returns:
        tuple[pathlib.Path, bool] : The path to the new (or existing) output 
            directory, and a bool indicating whether a new output directory was
            created. new directory=True, existing directory=False.

    Raises:
        FileNotFoundError : If unable to locate dataroot directory, or if
            unable to locate any shard files in existing output directory.
        RuntimeError : If unable to create new output directory for any reason.
            Will also raise the root exception.
    '''
    is_new = True
    dataroot = Path(dataroot)
    if not dataroot.exists():
        raise FileNotFoundError(
            f'Unable to locate dataset at path: {dataroot.as_posix()}')
    print(f'Dataset root found: {dataroot.as_posix()}\n'
          f'Locating output directory...')
    
    output_dir = Path(dataroot, f'tensor_cache_{cache_name}')
    if output_dir.exists():
        shards = list(output_dir.glob('*.pt'))
        if len(shards) == 0:
            raise FileNotFoundError(
                f'Unable to locate latent cache shards at path: '
                f'{output_dir.as_posix()}')
        print(f'Output directory found: {output_dir.as_posix()}\n'
              f'Shard count: {len(shards)}')
        is_new = False
    else:
        try: output_dir.mkdir()
        except Exception as e:
            raise RuntimeError(
                f'Unable to create output directory for tensor cache at path: '
                f'{output_dir.as_posix()}') from e      
    
    return output_dir, is_new

def print_progress(
        current_batch: int,
        num_batches: int
    ) -> None:
    sys.stdout.write('\x1b[1A')
    sys.stdout.write('\x1b[2K')

    progress = float(current_batch) / float(num_batches)
    progress_msg = (
        f'Progress: {current_batch} / {num_batches} | '
        f'{round(progress*100.0, 2)}% ')
    print(progress_msg)

