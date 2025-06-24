import argparse
from pathlib import Path
from typing import Any

from nectargan.testers.tester import Tester

def init_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--experiment_directory', type=str,
        help='The directory of the experiment to load for testing.')
    parser.add_argument(
        '-f', '--config_file', type=str, default=None,
        help='The system path to config file to use for testing.')
    parser.add_argument(
        '-d', '--dataroot', type=str, default=None,
        help='The path to the dataset root directory to use for testing.')
    parser.add_argument(
        '-l', '--load_epoch', type=int,
        help='The checkpoint epoch number to load for testing.')
    parser.add_argument(
        '-i', '--test_iterations', type=int, default=10, 
        help='The number of images from the `test` set to run the model on.')
    return parser.parse_args()

def validate_config(args: Any) -> str:
    if args.config_file is None:
        exp_dir = Path(args.experiment_directory)
        if not exp_dir.exists():
            raise FileNotFoundError(
                f'Unable to locate experiment directory: {exp_dir}')
        configs = sorted(list(exp_dir.glob('train*_config.json')))
        if len(configs) == 0:
            raise FileNotFoundError(
                f'No config files found in experiment directory: {exp_dir}\n'
                f'The "-f" flag can be used to set a config path explicitly.')
        return configs[-1]
    else: return args.config_file

def main():
    args = init_cli()
    config = validate_config(args=args)

    tester = Tester(
        config=config,
        experiment_dir=args.experiment_directory,
        dataroot=args.dataroot,
        load_epoch=args.load_epoch)
    tester.run_test(image_count=args.test_iterations)

if __name__== "__main__":
    main()