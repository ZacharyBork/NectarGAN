# NectarGAN Toolbox - Home
#### A graphical tool for training and testing models, reviewing results of previous tests, converting models to ONNX and testing the resulting model, and processing dataset files, all packaged into a single modern and easy to use interface.

> [!TIP]
> It is recommended to start with the [Toolbox quickstart guide](getting_started/toolbox_training.md), a short walkthrough which will teach you the basics of training and testing model, and reviewing training data using the NectarGAN Toolbox.

## Sections
**The Toolbox interface is broken down into seven core sections, each of which can be accessed via the buttons of the left-hand bar, or by pressing `Ctrl+[1-7]`. These sections are:**

### Experiment
Here you will find settings related to experiment output, naming, and versioning, as well and settings related to the architecture of the generator and the discriminator. There is also an option on this page to select and load a JSON config file and initialize all of the UI settings from the values defined in the file.

See the [NectarGAN Toolbox - Experiment](toolbox/experiment.md) documentation for more information.
### Dataset
In this section, you will find settings related to dataset file loading and training-time data augmentations.

See the [NectarGAN Toolbox - Dataset](toolbox/dataset.md) documentation for more information.
### Training
Here you will find settings related to model training including checkpoint loading, learning rate scheduling, loss function weighting, and checkpoint and example saving.

See the [NectarGAN Toolbox - Training](toolbox/training.md) documentation for more information.
### Testing
This section houses settings related to testing trained models and visualizing test results.

See the [NectarGAN Toolbox - Testing](toolbox/testing.md) documentation for more information.
### Review
The section allows you to review the results of previous training sessions, including the loading of post-epoch example images and graphing of loss log data.

See the [NectarGAN Toolbox - Review](toolbox/review.md) documentation for more information.
### Utilities
This section contains a number of additional tools that do not fit neatly into one of the above categories. These tools allow you to convert you models to ONNX and test the resulting `.onnx` model, and process dataset images in a variety of ways.

See the [NectarGAN Toolbox - Utilities](toolbox/utilities.md) documentation for more information.
### Settings
This section contains some general setting related to the Toolbox interface such as update rate during training and always-on-top behavior of the main Toolbox window.

See the [NectarGAN Toolbox - Settings](toolbox/settings.md) documentation for more information.

---
*Relevant sections:*
[Experiment](toolbox/experiment.md) | [Dataset](toolbox/dataset.md) | [Training](toolbox/training.md) | [Testing](toolbox/testing.md) | [Review](toolbox/review.md) | [Utilities](toolbox/utilities.md) | [Settings](toolbox/settings.md) |

###### NOTE: There is currently no developer documentation related to the NectarGAN Toolbox. It will be added in a future update. If you are curious how something behaves or would like to alter the functionality of the interface, the Toolbox source can be found at [`nectargan.toolbox`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/toolbox/), and is broken down into various submodules related to UI sections and/or functionality. Much of it is well documented in-line and in method docstrings.