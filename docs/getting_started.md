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
> NectarGAN requires a Python version >= `3.12`.

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
**The Toolbox quickstart guide is split in to three sections which are intended to be followed in order.**

**Section 1:** [*Training a Model*](/docs/getting_started/toolbox_training.md)<br>
**Section 2:** [*Testing a Model*](/docs/getting_started/toolbox_testing.md)<br>
**Section 3:** [*Reviewing Results*](/docs/getting_started/toolbox_training.md)

## NectarGAN CLI
**For more information regarding the CLI options currently offered by NectarGAN, please see the [*NectarGAN CLI quickstart*](/docs/getting_started/cli.md).**

## NectarGAN API
**Please see the [*API documentation*](/docs/api.md) for more information on getting started with the NectarGAN API.**

## Testing your installation
**If you would like to test whether your installation is working correctly,** NectarGAN includes a number of prebuilt tests which are run in different ways depending upon your installation type. See [here](/docs/testing_your_installation.md) for more information.

---