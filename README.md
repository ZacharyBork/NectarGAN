# NectarGAN
>A full-featured graphical development environment and model assembly framework for Pix2pix-style conditional GANs.

![_toolbox_training_gif_](/docs/resources/gifs/toolbox_training.gif)
## What is NectarGAN?
NectarGAN is comprised of two core components:

1. **The NectarGAN Toolbox.** A modern, high performance interface encapsulating a full end-to-end, production-ready cGAN pipeline. From assembling and managing Pix2pix-style datasets, to building and training conditional GAN models, to tracking and reviewing experiments, to testing trained models and preparing them for deployment. Your models can go from an idea to production, all without leaving the Toolbox.

    **Check out the [*Toolbox quickstart guide*](/docs/getting_started/toolbox_training.md) to get started.**

2. **The NectarGAN API.** A fully modular, highly extensible PyTorch-based framework comprised of easy to use building blocks for assembling, training, and testing conditional GAN models. The API offers developers the tools needed to quickly write cGAN training/testing scripts with maximum functionality and minimal boilerplate.

    By way of example, this is a complete (albeit relatively minimal) Pix2pix training script using the NectarGAN API:
    ```python
    from nectargan.trainers.pix2pix_trainer import Pix2pixTrainer

    if __name__ == "__main__":
        trainer = Pix2pixTrainer(
        config='/path/to/config/file.json', 
            loss_subspec='extended', log_losses=True)

        for epoch in range(100):
            trainer.train_paired(epoch) 
            trainer.save_checkpoint() 
            trainer.save_examples()
            trainer.print_end_of_epoch()
    ```
    > *The core [*NectarGAN paired training script*](/scripts/paired/train.py) is not much larger than this.*
    
    **For more information, please see the [*NectarGAN API documentation*](/docs/api.md).**
    
## Features
### [*Training Framework*](/docs/api.md)
- **Fully modular, PyTorch-based GAN training framework**.
- **Configurable [*UNet-style generator*](/docs/api/models/unet.md)** with support for multiple UNet block types, and drop-in support for your own custom blocks.
- **Easy loss tracking and logging** during and after training with a custom [*LossManager system*](/docs/api/losses/lossmanager.md).
- **Modular [*loss spec system*](/docs/api/losses/loss_spec.md)** allows you to quickly define reusable objective functions which can be loaded by the LossManager. Also included is a [*prebuilt Pix2pix objective function*](/nectargan/losses/pix2pix_objective.py) with a number of additional subspecs which add some extra loss functions for rapid prototyping.
- **The LossManager also supports a custom loss weight [*scheduling system*](/docs/api/scheduling.md)** allowing you to apply one of the included weight schedules, or to easily drop your own custom scheduling function in and have the LossManager take care of the rest.
- **Real time inference and loss visualization** during training with [*Visdom*](/docs/api/visdom.md) (for headless training), or with the NectarGAN Toolbox GUI.
- **Hook-based [*Trainer*](/docs/api/trainers/trainer.md) class** allows you to quickly define a custom training loop for your models.
- **Easy framework for loading pre-trained model weights** to experiment with new training schedules from a earlier checkpoint, or to fine tune a previously trained model. 
### [*Fully Config-Driven*](/docs/api/config.md)
- **Training and testing settings are all handled by [JSON configuration files](/docs/api/config.md#L7).**
- **Config copies are automatically exported and version tagged** for each train allowing for easy experiment tracking and reproducibility.
- **Toolbox UI settings can be loaded in one click from any previously saved config file** allowing you to easily pick up training with your previous settings.
- **Everything related to a training session is saved in the config file** ensuring that no experiment data is lost.
- **Automatic config parser allows you to easily [add entries](/docs/api/config.md#L65) to the config file** so that you can expand it to suit your model's needs.
### [*NectarGAN Toolbox*](/docs/toolbox.md)
- **Modern, fully graphical GAN training, testing, and evaluation tool.**
- **Configure your model architecture, build your dataset processing chain, define your objective function, and run your training, all with just a few clicks.** No code needed, just drag some sliders and see how it changes the models behavior.
- **Real time progress and timing stats, inference visualization, and loss graphing** with a highly configurable [*training interface*](/docs/getting_started/toolbox_training.md) allowing you to put the most focus on whatever metrics you're most interested in seeing.
- **A [*testing interface*](/docs/getting_started/toolbox_testing.md) allowing you to quickly load and validate your models on test datasets.** Define the epoch to load and how many images, and your model will be automatically run on a random selection of the provided images. The results will be evaluated with various loss functions and all the images and information will be displayed to you in the interface where you can sort by these metrics. Multiple tests can also be run and the results of each can be quickly swapped between directly from the UI.
- **Threaded training and testing framework** ensures that the UI remains responsive during training and that data visualization has a minimal impact on training and testing speed.
- **Training can easily be paused and resumed at any time.**
- **Load and visualize your model's training example images and loss log data** with the [*review panel*](/docs/getting_started/toolbox_review.md). All [*x, y, y_fake*] example sets will be automatically loaded, and each loss in the log will be graphed with a configurable sample rate.
- **View time statistics during training** including slowest, fastest, and average epoch and iteration time, total train time, and a real time time graph of all previous epoch times.
### [*Easy Dataset Augmentation*](/docs/toolbox/dataset.md)
**A simple but powerful augmentation UI** allows you to quickly apply a variety of Albumentations-based augmentations to your datasets at training-time to help expand small datasets and improve model generalization. Anything from random flipping and rotation, to optical distortion and random grayscale can be applied by just dragging a slider.
### [*ONNX Conversion and Testing*](/docs/toolbox/utilities.md#L5)
**Easily convert you models to ONNX** from either the toolbox UI, or from your own pipeline via the ONNXConverter, and test the resulting model immediately on real images with a live dashboard to display the results.
### [*Dataset Tools*](/docs/toolbox/utilities.md#L31)
**NectarGAN Toolbox also includes a set of helpful dataset processing tools. Currently, these tools allow you to:** 
- **Pair (A | B) images into Pix2pix input data**, extremely quickly, with control over direction and optional image scaling.
- **Sort image files by various metrics** (white pixel count, black pixel count, mean pixel value, various types of contrast), and then two related tools. One to unsort the files back to their original order, and one to copy to sorting order of one directory of files to another. These can be used to help process and find bad images in large datasets.
- **Automatically build train/test/val splits for dataset images** with easy control over split percentages.

## Getting Started
#### Please refer to the [*quickstart documentation*](/docs/getting_started.md) for information on how to get started using NectarGAN.

## Frequently Asked Questions
#### Please see [*here*](/docs/faq.md).

## Who is this for?
- Artists wanting to experiment with paired image translation tasks (why I first starting experimenting with Pix2pix).
- Reseachers who want a flexible environment in which to explore conditional GAN behavior.
- Startups looking for an easy and flexible way to experiment with and deploy image translation models.
- Students wishing to learn about and explore GANs in a visual, easy to use environment.
- Engineers/TDs wanting to painlessly integrate paired image translation models into their pre-existing pipelines.

## Project Status
**NectarGAN is under active development.** In its current state, however, it already offers:
- A robust framework for running and tracking experiments.
- An interactive dashboard for visualizing experiment results.
- A developer friendly API for constructing, training, and testing paired image-to-image adversarial models.
- An expansive and easy to use data augmentation pipeline and a variety of dataset processing tools.
- An interface to test your trained models, both as a `.pth` and as a `.onnx` to ensure consistency at deployment time.

**Planned future updates include:**
- Unpaired model assembly and training. More core trainer/tester classes.
- Multi-GPU support.
- More CLI support.
- Exposed normalization options in the Toolbox interface.
- More block types, standard loss functions, scheduling functions, and LossManager specifications.
- Live (and maybe interactive) architecture diagram.
- ONNX vs. PyTorch inference comparison.
- Loss monitoring and early stopping.
- Dataset augmentations preview.
- Houdini tooling. I'd like to build an HDA that lets you run NectarGAN training steps as PDG work items.

## Acknowledgements
This project is inspired by Jun-Yan Zhu's [*PyTorch-CycleGAN-and-pix2pix*](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) implementation.

## References
- [*Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks (Isola et al., 2017)*](https://arxiv.org/abs/1611.07004)

