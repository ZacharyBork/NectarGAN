from nectargan.start.torch_check import validate_torch
validate_torch()

from nectargan.trainers.diffusion_trainer import DiffusionTrainer

def main():
    trainer = DiffusionTrainer(
        config='/media/zach/UE/ML/NectarGAN/nectargan/config/default.json', 
        log_losses=True)

    epoch_count = 100
    for epoch in range(epoch_count):
        trainer.train_diffusion(epoch) 
        trainer.print_end_of_epoch()

if __name__ == "__main__":
    main()

