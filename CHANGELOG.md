# Changelog

**All notable changes to this project will be documented in this file.**

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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


