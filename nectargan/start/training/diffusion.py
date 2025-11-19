from nectargan.start.torch_check import validate_torch
validate_torch()

from pathlib import Path

from nectargan.trainers.diffusion_trainer import DiffusionTrainer

def get_diffusion_config() -> Path:
    root = Path(__file__).parent.parent.parent
    diff_config = Path(root, 'config/defaults/diffusion.json')
    if not diff_config.exists():
        raise FileNotFoundError(
            f'Unable to locate default diffusion config at path: '
            f'{diff_config.as_posix}')
    return diff_config

def main():
    config = get_diffusion_config()
    trainer = DiffusionTrainer(config=config.as_posix(), log_losses=True)

    epoch_count = 100
    for epoch in range(epoch_count):
        trainer.train_diffusion(epoch) 
        trainer.print_end_of_epoch()

if __name__ == "__main__":
    main()

