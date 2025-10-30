# NectarGAN – Frequently Asked Questions (FAQ)

>  ## Sections
> 1. [Installation & Environment](#installation--environment)
> 2. [Dataset & Configuration](#dataset--configuration)
> 3. [Training (Toolbox & CLI)](#training-toolbox--cli)
> 4. [Toolbox Overview](#toolbox-overview)
> 5. [Testing & Reviewing Results](#testing--reviewing-results)
> 6. [ONNX Export & Deployment](#onnx-export--deployment)
> 7. [Scheduling, Losses & Internals](#scheduling-losses--internals)
> 8. [Troubleshooting](#troubleshooting)

## Installation & Environment

### 1) What platforms are supported?
NectarGAN has been tested on Windows. Linux support is planned. See the [Getting Started](docs/getting_started.md#L4) page for more info.

### 2) What Python versions are supported?
Python >= 3.12

### 3) How do I install it?
- Clone the repo: `git clone https://github.com/ZacharyBork/NectarGAN.git`
- Create a fresh environment and install:
```bash
pip install .

--- or ---

conda env create -f environment.yml
```
Dev/testing install:
```bash
pip install -e ".[dev]"

-- or --

pip install -r requirements.dev.txt
```
### 4) How can I verify my installation works?
Run the test suite with pytest. See [here](/docs/testing_your_installation.md) for more information.

### 5) I tried running the Toolbox/CLI, but it says I'm missing PyTorch?
PyTorch is not included in the core dependencies. It must be installed separately. PyTorch has multiple versions, each tied to a compute platform, and you should choose the one that best suits your needs based on the system you are running NectarGAN on. Installation instructions for PyTorch can be found [on their website](https://pytorch.org/get-started/locally/).

> [!NOTE]
> At this time, NectarGAN has been tested on the CUDA 12.6, CUDA 12.8, and CPU compute platforms.

## Dataset & Configuration

### 1) What dataset layout does paired training expect?
Your root dataset directory should be laid out as follows:
```
root/
├─ train/
├─ val/
├─ test/
```
> [!NOTE]
> The `test` directory is optional. The dataset images in this directory can be used in a separate post-train model validation. See [here](/docs/getting_started/toolbox_testing.md).

### 2) How do I set/override config values?
**Toolbox:** The Toolbox generates training configs dynamically. When using it, all config values can be set via the UI. See [here](/docs/getting_started/toolbox_training.md).

**CLI:** The default config file used by the training scripts can be found at [`/nectargan/config/default.json`](/nectargan/config/default.json). You can open this file in any text editor to override values. See [here](/docs/api/config.md) for more information.

**Custom Training Script:** There are a number of ways to manage configuration options when writing a custom training script. See [here](/docs/api/config.md) for more information. 

### 3) Can I train on grayscale or multi-channel images?
This can be controlled by setting the value of `in_channels` on the [UNet](/docs/api/models/unet.md) and [PatchGAN](/docs/api/models/patchgan.md) models. 1 for grayscale, 3 for RGB. I haven't tested multi-channel inputs much; you may encounter problems. I will check on this and fix it if necessary at some point in the future. ONNX model conversion currently only supports 3 channels. See [here](/docs/toolbox/utilities.md) and [here](/docs/api/onnx_tools.md) for more information on ONNX conversion.

### 4) How do I select the mapping direction (A->B vs B->A)?
By setting the value of `config.dataloader.direction` in your config file. Valid values are `AtoB` and `BtoA`. See [here](docs/api/dataset.md) for more information regarding dataset loading.

## Training (Toolbox & CLI)

### 1) How do I start Pix2pix training from the CLI?
See [here](docs/getting_started/cli.md).

### 2) What loss setups are available?
There are 4 pre-build loss subspecs for the Pix2pix model objective. These are:
- `basic`
- `basic+vgg` (adds VGG19 perceptual)
- `extended` (adds L2, Sobel, Laplacian)
- `extended+vgg`

See the [documentation on loss functions](docs/api/losses/loss_functions.md) and the [Pix2pix objective function](/nectargan/losses/pix2pix_objective.py) for more information on the behaviour of these loss subspecs.

### 3) How can I control loss weights?
**Toolbox:** On the Training panel, under the loss tab, you will find sliders for the various loss functions.

**CLI:** In the config file, under `config.train.loss`, you can set weights for the various loss functions by changing the correspoding value.

**Custom Training Script:** See the [LossManager](/docs/api/losses/lossmanager.md) documentation for information on setting loss weights and applying weight schedules.

### 4) Can I resume training from a checkpoint?
**Toolbox:** Yes. On the Training tab, check the `Continue Train` checkbox and input the epoch you would like to load. When training begins, Toolbox will look for the config file for the given epoch in the experiment directory and load it, if present, to continue training.

**CLI:** Yes. Just use the `-f` `--config_file` flag and pass it the path the the config file for the epoch you would like to resume training from. See the [CLI documentation](/docs/getting_started/cli.md) for more information.

**Custom Training Script:** See [here](/docs/api/trainers/building_a_trainer.md).

### 5) How are learning rates scheduled?
**Toolbox:** On the Training panel, you can set a learning rate schedule with the `Epochs`, `Epochs Decay`, and `Initial` and `Target` learning rate. See [here](/docs/api/scheduling/schedule_functions.md) for more info. 

**CLI:** `Epochs`, `Epochs Decay`, and `Initial` and `Target` learning rate can all be set in the config file, under `config.train.generator.learning_rate` and `config.train.discriminator.learning_rate`.

**Custom Training Script:** See [here](/docs/api/scheduling.md) for more information on scheduling.

### 6) What input resolution can I use?
Any reasonable resolution. Higher resolutions take longer to train and are oftentimes more difficult to train in a stable manner. They also require more memory which is ultimately the limiting factor. Individual training images (not paired), should have a 1:1 aspect ratio, and a power of two resolution. Popular choices are `256x256` and `512x512`. Sometimes higher if batch size is kept low. 

Just be sure to use a reasonable number of layers for your chosen resolution. Higher resolutions generally require a higher layer count for both generator and discriminator to produce acceptable results. But too many layers will cause training to fail, as the tensor will be downsampled too far before hitting the bottleneck.

See [here](/docs/api/models/unet.md#L34) for more information.

### 7) I’m seeing checkerboard artifacts. How can I reduce them?
These artifacts are incredibly common with pixel to pixel GAN models. Sometimes training longer will help, or just increasing the decay epochs to give the model longer to settle. You can also try using the transposed convolution upsampling method for the generator (see [here](docs/toolbox/experiment.md#L13)), or using the [Residual UNet block](/docs/api/models/unet_blocks.md#L27) instead of the standar UNet block. Both of these solutions are frequently able to reduce or completely remove the checkerboarding.

If none of the above solutions work, you may also try adjusting your loss values. Sometimes adding L2 or VGG Perceptual, and/or reducing L1, can also help to eliminate this artifacting. 

## Toolbox Overview

### 1) What are the Toolbox sections and shortcuts?
See [here](/docs/toolbox.md).

### 2) Where do outputs go and how are experiments versioned?
Experiments will be exported to the `Output Root`, in directories named `Experiment Name`. These directories are versioned automatically (see [here](/docs/toolbox/experiment.md)). **`Output Root` and `Experiment Name` are set in different places depending upon how training is initiated:**

**Toolbox:** Both can be set on the Experiment panel.

**CLI:** Both are set in the config file under `config.common`.

**Custom Training Script:** Same as CLI.

### 3) Can I change UI performance vs feedback rate?
Yes. When training with Toolbox, the actual training happens in a separate thread from UI, and that thread sends updates back to the UI every so often. Doing this too frequently is performance intensive and slows down training significantly. You could have it send back an update every iteration, though, if you wanted. 

This update frequency can be changed by going to the Settings panel and setting the value of `Training Update Rate`. **This value can be changed at runtime,** so you can decrease it briefly to get a better look at the models output, then increase it again to revert to your initial training speed.

## Testing & Reviewing Results

### 1) How do I test a trained model?
**Toolbox:** Using the Testing panel. See [here](/docs/getting_started/toolbox_testing.md)

**CLI:** See [here](/docs/getting_started/cli.md) for information on model testing via CLI.

**Custom Testing Script:** See the [Tester class documentation](/docs/api/testers.md).

### 2) How do I review my experiments?
The Toolbox Review panel offers an easy way to review the results of your model test. See [here](/docs/getting_started/toolbox_review.md) and [here](/docs/toolbox/review.md) for more information.

### 3) How do I log losses during training?
**Toolbox:** On the Training panel, under the Loss tab, you can enable `Log Losses During Training` and configure the logging behaviour.

**CLI:** Using the `--log_losses` flag. See [here](/docs/getting_started/cli.md).

**Custom Training Script:** See the [LossManager documentation](/docs/api/losses/lossmanager.md).

## ONNX Export & Deployment

### 1) How do I export to ONNX?
**Toolbox:** On the Utilities panel, you will find a set of tools that allow you to convert your models to run on the ONNX runtime, and to test your converted models. See [here](/docs/toolbox/utilities.md) for more information. 

**CLI:** This is currently not supported.

**Custom Script:** See the [ONNX tools documentation](/docs/api/onnx_tools.md).

### 2) Why do I get instance-norm warnings when exporting?
See [here](/docs/toolbox/utilities.md#L7).

### 3) Can I test an exported ONNX model inside the Toolbox?
Yes, see [here](/docs/toolbox/utilities.md).

### 4) Does ONNX export support non-RGB inputs?
No, currently the ONNXConverter only supports 3 channels (RGB).

## Scheduling, Losses & Internals

### 1) What is the LossManager and why use it?
The Loss Manager is a comprehensive module for managing everything related to model loss during training. It is highly flexible and configurable, and allows you to easily and accurately manage complex objective functions.

See the [Loss Manager documentation](/docs/api/losses/lossmanager.md) for more information.

### 2) What are “loss specs”?
Loss specs are drop in objective functions which you can pre-define, and feed in to a Loss Manager, allowing you to more carefully build, track, and reuse objectives from model to model.

See the [Loss Spec documentation](/docs/api/losses/loss_spec.md) for more information, and the [Pix2pix Objective Function](/nectargan/losses/pix2pix_objective.py) for an example of a loss spec.

### 3) How do schedules integrate with training?
NectarGAN offers a generic [Scheduler](/docs/api/scheduling/schedulers.md#6), and a wrapper around the native PyTorch learning rate scheduler called [TorchScheduler](/docs/api/scheduling/schedulers.md#51). This allows you to use the same [Schedule Functions](/docs/api/scheduling/schedule_functions.md) for each. 

The TorchScheduler is predominantly used for learning rate, to take advantage of the inherant integration with the PyTorch optimizers. The generic Scheduler is predominantly used for loss weight scheduling, though you could use it for whatever you want in your own models.

See [here](/docs/api/scheduling.md) to get started with scheduling in NectarGAN.

## Troubleshooting

### 1) Training is slow in the Toolbox. Any tips?
Increase Training Update Rate (see [here](/docs/toolbox/settings.md)), and avoid very low dump frequencies for loss logs, as these can potentially cause lag spikes.

### 2) I can’t find my outputs/checkpoints. Where are they?
All files related to a given training session will be exported to your `Output Root` directory, in a subdirectory named after your current `Experiment Name`. These directories will be automatically versioned, so be sure to look for the latest one.

### 3) CLI won’t see my config. What’s used by default?
If the `-f` flag isn't used, the training/testing scripts will instead use the default config file located at [`/nectargan/config/default.json`](/nectargan/config/default.json).

