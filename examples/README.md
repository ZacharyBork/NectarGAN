# NectarGAN - Examples
**This directory contains examples of models trained in various configurations.**

## Example Types
**Examples are grouped in to type based on their convolutional block type and by the loss specification the example was run with.**

These are the current examples:

- UNet (UnetBlock)
    - Basic
    - Basic + VGG
    - Extended
- ResUNet (ResidualUnetBlock)
    - Basic
    - Basic + VGG

**Below are the loss weights for each example type:**
| Example Type | GAN | L1 | L2 | Sobel | Laplacian | VGG |
| --- | --- | --- | --- | --- | --- | --- |
| Basic | 1.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Basic + VGG | 1.0 | 100.0 | 0.0 | 0.0 | 0.0 | 10.0 |
| Extended | 1.0 | 100.0 | 0.0 | 3.0 | 1.0 | 10.0 |

## Steps to Reproduce
**Each example directory includes the training config file used to create the example.** To reproduce the results seen in the example, follow the steps below:

1. **Follow the steps in the [Getting Started](https://github.com/ZacharyBork/NectarGAN/blob/main/docs/getting_started.md) guide** to install NectarGAN on your system.
2. **Use the Pix2pix dataset [download script](https://github.com/ZacharyBork/NectarGAN/blob/main/docs/scripts.md)** to download the `facades` dataset used for the examples.
3. **Open the `examples` directory, and select which example you would like to reproduce.**
4. **From the selected example directory, open the `config.json` file in a text editor.** Then set the value of these three keys:

| Key | Expected Path |
| --- | --- |
| `config.common.device.` | Either `"cpu"` or `"cuda"`, based on the PyTorch platform version you installed in step 1. |
| `config.common.output_directory` | The directory on your computer where you would like the experiment output to go. A named and versioned subdirectory will be created for the actual experiment. |
| `config.dataloader.dataroot` | The path to the dataset directory to use for training. If you downloaded the dataset in step two, this will be: `/path_to_NectarGAN/data/datasets/facades` |

5. **Save the file,** open a terminal in the environment you installed NectarGAN in, navigate to the NectarGAN root directory, and run this command to begin training:
```bash
python -m nectargan.start.training.paired -f {config_path} -lss {subspec} -log
```
**Replace `{config_path}` with the actual path to your chosen example `config.json` on your system. `{subspec}` should be the chosen subspec for your example type:**

| Example Type | -lss |
| --- | --- |
| Basic | `"basic"` |
| Basic + VGG | `"basic+vgg"` |
| Extended | `"extended"` |
| Extended + VGG | `"extended+vgg"` |

*Note: Subspecs which include a `+vgg` will automatically download the VGG19 default weights into the Python environment the script is being run from if you do not already have them downloaded.*

**And that's it!** 

If you now open a web browser and visit `http://localhost:8097`, you should see visdom updating with runtime loss graphs and training examples. If you visit the directory you set for `output_directory` in step 4, you should also see a new folder was created, named after your chosen example, which contains loss logs and a copy of the training config. Then after each epoch, you will see a set of example images exported to the `examples` subdirectory, and every five epochs, a model checkpoint will also be saved to the `output_directory`.
