# NectarGAN - Getting Started (CLI)
#### This section will walk you through the process of using NectarGAN from the command line.

> [!NOTE]
> Up to this point, my focus has primarily been on the Toolbox interface. As such, the CLI options currently are rather limited. What does exist is really just a slightly expanded version of the scripts I originally used for validation of the Pix2pix implementation.
>
> **Expanding CLI support is a high-priority task, so this is likely to change significantly in the near future.**

## Training
#### Reference: [`nectargan.start.training.paired`](/nectargan/start/training/paired.py)
**Pix2pix training can be performed via CLI in NectarGAN with the following command:**
```bash
nectargan-train-paired -f /path/to/config.json -lss extended+vgg --log_losses
```
There are currently only three command line arguments, all of which are shown here. Following is a brief explanation of each:
### `-f` `--config_file`
**Type:** `string` *(optional)*<br> 
**Description:** The system path to the config file to use for training. If this flag is not set, the default config (`/nectargan/config/default.config`) will be used.
### `-lss` `--loss_subspec`
> [!IMPORTANT]
> The first time you run a loss subspec which includes `+vgg`, the VGG19 default weights (`IMAGENET1K_V1`) will be downloaded from PyTorch if you have not previously downloaded them in the Python environment which you are running the command from.

**Type:** `string` *(optional)*<br>
**Description:** What loss subspec to use when initializing the objective function. Currently, the valid `loss_subspecs` are:
1. `basic`: (default) Objective function as described in [Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks*, CVPR 2017.](https://arxiv.org/abs/1611.07004)
2. `basic+vgg`: `basic`, but with added VGG19-based perceptual loss (see [here](/nectargan/losses/pix2pix_objective.py#L56)).
3. `extended`: `basic`, but with a few added loss functions:
    - Mean squared error (L2)
    - Sobel (large-scale structure, see [here](/docs/api/losses/loss_functions.md#L4))
    - Laplacian (small-scale structure, see [here](/docs/api/losses/loss_functions.md#L22))
4. `extended+vgg`: `extended`, but with added VGG19-based perceptual loss.

*See [`pix2pix_objective`](/nectargan/losses/pix2pix_objective.py) for more info.*
### `-log` `--log_losses`
**Type:** `action`, `store_true` *(optional)*<br>
**Description:** If this flag is present, the [`LossManager`](/docs/api/losses/lossmanager.md) will be set up to cache loss values and occasionally dump the cache to the loss log.
### `--help`
**Description:** Shows information about the script's arguments.

### Summary
**So, in short, all the actual training data is still pulled from the config file (see [here](/docs/api/config.md)), as with any other type of training in NectarGAN.** This means you could duplicate the default config, set it up however you'd like, then just point the train script to it via the `-f` flag when running the `nectargan-train-paired`. 

**It also keeps the actual CLI simple** which I generally prefer, as now, only things which aren't present in the config, like `loss_subspec` (since it's just a drop-in objective function, adding it's parameters to the config is not necessary), or things which you might want to override, need to have CLI arguments.

> [!NOTE]
> **When training via CLI rather than the Toolbox interface, you will likely want to use the alternative training visualization method, Visdom.**
>
> This is very simple, just:
> 1. Open the config file you will be using for training 
> 2. Find the entry for `config`/`visualizer`/`visdom`.
> 3. Under it, you will see a boolean value called `enable`. Set this to `true`. 
>
>The rest of the settings in that section can be used to configure the behavior of the Visdom-based visualizer.
>
> *See [here](/docs/api/visdom.md) for more information.*

## Testing
#### Reference: [`nectargan.start.testing.paired`](/nectargan/start/testing/paired.py)
**Pix2pix model testing can be performed via CLI in NectarGAN with the following command:**
```bash
nectargan-test-paired -e /path/to/experiment/directory -l 200 -d /path/to/dataset/root -i 30
```
The testing script has five arguments, two of which are required with the rest being optional:
### `-e` `--experiment_directory`
**Type:** `string` *(required)*<br>
**Description:** The directory of the experiment to load for testing.

### `-l` `--load_epoch`
**Type:** `int` *(required)*<br>
**Description:** The checkpoint epoch number to load for testing (i.e. `epoch{load_epoch}_netG.pth.tar`).

### `-f` `--config_file`
**Type:** `string` *(optional)*<br>
**Description:** The system path to config file to use for testing. If this is not set, the testing script will first look for the latest training config in the given experiment directory (i.e. `train{x}_config.json`), which is, by default, exported automatically at the start of each training session. If this is set, it will use this config file instead.

### `-d` `--dataroot`
**Type:** `string` *(optional)*<br>
**Description:** The path to the dataset root directory to use for testing. If this is not set, the dataset root specified in the config file (whether that be the automatically selected one, or the one defined by `--config_file`) will be used for testing.

> [!NOTE]
> The given dataset root must have a `test` subdirectory containing dataset images for testing!

### `-i` `--test_iterations`
**Type:** `int` *(optional)*<br>
**Description:** The number of images from the `test` set to run the model on. If not set, the default `10` will be used. The test script will grab this number of random images from the `test` set, run the model inference on them, and export the results to a `test` subdirectory inside of the given experiment directory.

### `--help`
**Description:** Shows information about the script's arguments.

## Utils
#### Reference: [`nectargan.utils.rebuild_default_config`](/nectargan/utils/rebuild_default_config.py)
Currently, there is only one utility accessible via CLI: 
```bash
nectargan-utils-restoreconfig
```
This utility will restore the default config file, with all of its original settings, at its original location (`/nectargan/config/default.json`). This can be used in the event that the origin config file gets lost, deleted, or otherwise broken beyond repair.

By default, the util will not allow overwriting of an existing default config, and will instead raise an exception if the file exists. However, you may use the [`-o`, `--overwrite_existing`] flag to allow it to overwrite the existing file.

---