import argparse

from nectargan.latent.latent_manager import LatentManager

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r', '--dataroot', type=str, 
        help=(
            'The root directory containing the images to build the latent '
            'tensors from.'))
    parser.add_argument(
        '-d', '--device', type=str, default='cuda', 
        help='The PyTorch device to use when encoding latent tensors.')
    parser.add_argument(
        '-t', '--dtype', type=str, default='float16', 
        help='The dtype of the latent space tensors.')
    parser.add_argument(
        '-m', '--model', type=str, default='stabilityai/sd-vae-ft-ema', 
        help='The VAE model to use for encoding.')
    parser.add_argument(
        '-s', '--latent_spatial_size', type=int, default=64, 
        help='The desired resolution (^2) of the latent-space tensors.')
    parser.add_argument(
        '-bs', '--batch_size', type=int, default=32, 
        help='The batch size to use for encoding and caching.')
    parser.add_argument(
        '-ss', '--shard_size', type=int, default=4096, 
        help='The desired size of each shard (in batches).')
    parser.add_argument(
        '-w', '--num_workers', type=int, default=0, 
        help='The desired resolution (^2) of the latent-space tensors.')
    parser.add_argument(
        '--store_file_names', action='store_true', 
        help=(
            'Whether to save the file names of the original images in the '
            'cache manifest. Used for metadata lookup during training.'))
    parser.add_argument(
        '--validate_cache', action='store_true', 
        help=(
            'If this flag is present, after the cache is exported, each shard '
            'will be loaded and checked for inf, NaN, etc.'))
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    manager = LatentManager(
        dataroot=args.dataroot,
        device=args.device,
        dtype=args.dtype,
        model=args.model,
        latent_spatial_size=args.latent_spatial_size,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        num_workers=args.num_workers,
        store_file_names=args.store_file_names)
    manager.cache(validate_cache=args.validate_cache)
    
    
