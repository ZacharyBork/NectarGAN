# NectarGAN - Getting Started
#### This document will guide you through the process of installing NectarGAN, and get you started using the NectarGAN Toolbox, training and testing models from the command line, and interacting with the NectarGAN API.

> [!IMPORTANT]
> Currently, NectarGAN has only been tested on Windows. Linux validation is planned for the near future and with it will come Linux-specific install instructions.
> 
> *This is high on the list of planned features but if you want to give it a try before I get to it, please do let me know what your experience is like.*
## Installing NectarGAN
### Windows
> [!TIP]
> It is recommended to install NectarGAN in a fresh environment to avoid dependency conflicts, especially as it is still in active development. 
>
> NectarGAN requires a Python version >= `3.11`.

1. **Clone the repository:** `git clone https://github.com/ZacharyBork/NectarGAN.git`
2. **Navigate to the repository root and run:**
- **Pip:**
```bash
pip install .
```
- **Anaconda:**
```bash
conda env create -f environment.yml
```
3. **Install your preferred version of PyTorch:**
> ###### From: https://pytorch.org/get-started/locally/
| Version | Command |
| :---: | --- |
**CPU-Only** | `pip install torch torchvision torchaudio`
**CUDA 11.8** | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
**CUDA 12.6** | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`
**CUDA 12.8** | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`

## NectarGAN Toolbox
**For information on getting started with the NectarGAN Toolbox, please see [here](/docs/getting_started/toolbox_training.md).**

## NectarGAN CLI

## NectarGAN API
