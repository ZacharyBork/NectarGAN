from nectargan.start.torch_check import validate_torch
validate_torch()

import argparse
from nectargan.testers.diffusion import DiffusionTester

def init_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--experiment_directory', type=str,
        help='The directory of the experiment to load for testing.')
    parser.add_argument(
        '-f', '--config_file', type=str, default=None,
        help='The system path to config file to use for testing.')
    parser.add_argument(
        '-l', '--load_step', type=int,
        help='The checkpoint step number to load for testing.')
    parser.add_argument(
        '-i', '--test_iterations', type=int, default=1, 
        help='The number of images from the `test` set to run the model on.')
    parser.add_argument(
        '-b', '--batches', type=int, default=1, 
        help='The number of batches to run per test iteration.')
    parser.add_argument(
        '-s', '--latent_spatial_size', type=int, default=64, 
        help='The resolution (^2) of the input latent tensors for inference.')
    parser.add_argument(
        '-m', '--inference_mode', type=str, default='DDIM', 
        choices=['DDIM', 'DDPM'], help='What sampling algorithm to use.')
    parser.add_argument(
        '-is', '--inference_steps', type=int, default=1000, 
        help='How many denoising steps to use for inference.')
    parser.add_argument(
        '-c', '--caption', type=str, default=None, 
        help='The caption to use for sampling.')
    parser.add_argument(
        '-cfg', '--cfg_scale', type=float, default=7.5, 
        help='The classifier free guidance scale to use for sampling.')
    parser.add_argument(
        '-ema', '--sample_ema', action='store_true', 
        help='Whether to sample from EMA when testing model inference.')
    return parser.parse_args()

def main():
    args = init_cli()

    tester = DiffusionTester(
        experiment_directory=args.experiment_directory,
        config_file=args.config_file,
        load_step=args.load_step,
        latent_size=args.latent_spatial_size,
        inference_mode=args.inference_mode,
        inference_steps=args.inference_steps,
        sample_ema=args.sample_ema)
    tester.run_test(
        image_count=args.test_iterations, batches=args.batches,
        caption=args.caption, cfg_scale=args.cfg_scale)

if __name__== "__main__":
    main()