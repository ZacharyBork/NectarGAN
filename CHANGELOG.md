# Changelog

**All notable changes to this project will be documented in this file.**

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0](https://github.com/ZacharyBork/NectarGAN/releases/tag/v0.3.0) - 2026-02-09

### Highlights
- Complete diffusion model implementation (DDPM/DDIM, pixel and latent space)
- Self-attention UNet with text conditioning support
- Loss masking framework for conditional GANs
- ResNet generator and classifier networks
- Multi-scale PatchGAN discriminator

### Added
- Top-level imports for some core API submodules
- ResNet generator network
- ResNet 50, 101, and 152 classifier networks
- Loss masking framework for paired conditional GANs
- Multi-scale PatchGAN discriminator
- Pixel-space (DDPM) diffusion model
- Latent-space (DDPM, DDIM) diffusion model
- New config JSON/dataclasses for diffusion models
- New Trainer and Tester classes for diffusion model
- Visdom visualizer class for diffusion models
- EMA support and running average loss tracker for diffusion models (will be added to cGAN models in future release)
- Self-attention UNet for diffusion noise prediction
- New cross attention UNet blocks for timestep and context embedding
- Wrapper for pre-trained text embedding models
- Framework for caption loading (with standarized metadata format)
- Tools for caption conversion for common diffusion datasets
- Wrapper for pre-trained variational autoencoder models
- Latent tensor pre-caching/cache loading framework
- CLI script for latent tensor caching
- Base for annotation creator tool. Currently non-functional, will be updated in a future release
- New base dataset class to serve as common parent for subclasses
- New pre-built datasets:
    - DiffusionDataset / ImageTextDataset
    - LAION streaming dataset
    - CacheLoader / CacheShardHotLoader
    - PairedMaskDataset
- Checkpointing for base UnetGenerator class
- Generator and Discriminator checkpointing for Pix2pix model
- New base classes for custom loss functions
- Utility script to retrieve config objects directly from file paths
- More control over cGAN ColorJitter from config file
- New augmentation application methods for unpaired and loss-masked training

### Fixed
- Race condition from image/loss queues when using Visdom visualizer in multi-threaded mode 
- Bug where Pix2pix loss functions were getting called with zero weight when not actively registered
- Memory fragmentation bug from mistimed CUDA synchronization and cache flushing
- Bug causing current epoch to be offset by one and total epoch count to be incorrect when loading Pix2pix model from checkpoint. LR scheduling should always return correct values now

### Changed
- Updated config file structure for paired cGAN models
- Loss management and scheduling models now use generic timesteps
- LossManager now allows dataset length of None (for cache hotloading)
- Layer weights for VGGPerceptual loss can now be controlled independently (API only for now)
- Base Visdom visualizer class now splits tensors by batch for visualization
- Config rebuild script now pulls definition directly from GitHub
- Most datasets no longer require Configs for init. The rest will follow in a future update
- LossManager no longer builds dummy tensors for previous losses. They are now build on the fly, as required
- Config JSONs now include a config_type variable to define their associated model type
- Config dataclasses now inherit from common parent, track their default JSON files, and also include a human-readable group schema for validation
- CUDA benchmarking, determinism, and fp32 precision can now be controlled from config files
- ColorJitter can now be applied to either input-only, or input+target for cGAN models

## [0.2.1](https://github.com/ZacharyBork/NectarGAN/releases/tag/v0.2.1) - 2025-11-07

### Added
- Helper script to graph loss data
- Helper script to combine {x, y_fake, y} example images from training
- UNet and ResUNet example images and loss graphs

### Fixed
- Bug in example image export that was reducing quality of exported example images during training
- Issue where test images would have augmentations applied to them during model validation

### Changed
- dataloader.load_size in all default config files from 256 -> 286

## [0.2.0](https://github.com/ZacharyBork/NectarGAN/releases/tag/v0.2.0) - 2025-11-04

### Added
- Mkdocs config
- Mkdocs site build workflow
- Docker deployment setup
- Docker build workflow
- Docker quickstart guide
- New CLI wrapper for Docker container

### Changed
- VisdomVisualizer class now takes an server endpoint argument. Defaults to `http://localhost` so default functionality is not affected

## [0.1.0](https://github.com/ZacharyBork/NectarGAN/releases/tag/v0.1.0) - 2025-10-31

### Added
- Graphical training, testing, and validation tool
- Modular GAN API
- Hook-based training loops
- Runtime loss management and logging framework
- Scheduling framework
- Native tooling for deploying models on the ONNX runtime


