# NectarGAN API - Home
#### The NectarGAN API is a fully modular, highly extensible PyTorch-based framework comprised of easy to use building blocks for assembling, training, and testing conditional GAN models. 

##### The API documentation is broken down into a number of sections:

## Config
At the core of the NectarGAN API's functionality is the JSON-based configuration system. Config files are parsed by the [`ConfigManager`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/config/config_manager.py) into a set of [`Config`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/config/config_data.py) dataclasses, allowing for easy access to any config value anywhere you need it. The configuration system is also at the core of the experiment tracking mechanisms in the NectarGAN API.

**For more information, please see [here](api/config.md)**
## Models
The API provides a modular UNet-style [generator](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/models/unet/model.py) and PatchGAN [discriminator](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/models/patchgan/model.py) model, both of which allow you to drop in your own convolutional blocks if you so choose.

**For more information about the UNet model, please see [here](api/models/unet.md).**<br>
**For more information about the PatchGAN model, please see [here](api/models/patchgan.md)**
## Dataset
The API provides a helper class called [`PairedDataset`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/dataset/paired_dataset.py) for loading dataset data for paired images translation models. These is also a helper class called [`Transformer`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/dataset/transformer.py) for performing Albumentations-based dataset augmentations.

**For more information, please see [here](api/dataset.md).**
## Trainers
The API's [`Trainer`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/trainers/trainer.py) class is an inheritable baseline class for training GAN models with a overrideable training methods and a number of helper functions for speeding up model creation. The [Pix2pixTrainer](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/trainers/pix2pix_trainer.py) serves as an example implementation of a Pix2pix-style[^1] cGAN model using the `Trainer` class as a base.

**For more information, please see [here](api/trainers/trainer.md).**
## Testers
The API's [`Tester`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/testers/tester.py) class allows you to test your trained model on a test dataset.

**For more information, please see [here](api/testers.md).**

## Losses
The API includes a robust and hightly extensible loss tracking and management system, the core of which is the [`LossManager`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/losses/loss_manager.py), a drop-in solution for handling all things related to loss in your models.

**For more information about the `LossManager`, please see [here](api/losses/lossmanager.md).**<br>
**For information about loss specs, simple functions that can be used to define reuseable objective functions in combination with the `LossManager`, please see [here](api/losses/loss_spec.md).**<br>
**For more information about the loss Modules included with the NectarGAN API, please see [here](api/losses/loss_functions.md).**

## Scheduling
The API has a simple schedule wrapper, allowing you to define schedules as Python functions and apply them to learning rates and loss weights.

**For more information, please see [here](api/scheduling.md).**

## Visdom
The API includes a utility class called [`VisdomVisualizer`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/visualizer/visdom/visualizer.py) for live loss graphing and visualization of [x, y_fake, y] image sets during training via [Visdom](https://github.com/fossasia/visdom).

**For more information, please see [here](api/visdom.md).**

## ONNX Tools
The API includes a simple ONNX converter class and model tester script for convenience. 

**For more information, please see [here](api/onnx_tools.md).**

[^1]: [Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks (Isola et al., 2017)](https://arxiv.org/abs/1611.07004)