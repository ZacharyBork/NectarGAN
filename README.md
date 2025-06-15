# NectarGAN

>NectarGAN is a graphical GAN development environment and model assembly framework for Pix2pix-style conditional generative adversarial networks.

## Features
### Training Framework
- **Fully modular, PyTorch-based GAN training framework**.
- **Configurable Unet-style generator** with support for multiple UNet block types, and drop-in support for your own custom blocks.
- **Easy loss tracking and logging** during and after training with a custom LossManager system.
- **Modular loss spec system** allows you to quickly define reusable objective functions which can be loaded by the LossManager. Also included is a prebuilt Pix2pix objective function with a number of additional subspecs which add some extra additional loss functions for rapid prototyping.
- **The LossManager also supports a custom loss weight scheduling system** allowing you to apply one of the included weight schedules, or to easily drop your own custom scheduling function in and have the LossManager take care of the rest.
- **Real time inference and loss visualization** during training with Visdom (for headless training), or with the NectarGAN Toolbox GUI.
- **Hook-based Trainer class** allows you to quickly define a custom training loop for your models.
- **Easy framework for loading pre-trained model weights** to experiment with new training schedules from a earlier checkpoint, or to fine tune a previously trained model. 
### Config-Driven Framework
- **Training and testing settings are all handled by JSON configuration files.**
- **Config copies are automatically exported and version tagged** for each train allowing for easy experiment tracking and reproducibility.
- **Toolbox UI settings can be loaded in one click from any previously saved config file** allowing you to easily pick up training with your previous settings.
- **Everything related to a training session is saved in the config file** ensuring that no experiment data is lost.
- **Automatic config parser allows you to easily add entries to the config file** (and in the future, to the UI as well) so that you can expand it to suit your model's needs.
### NectarGAN Toolbox
- **Modern, fully graphical GAN training, testing, and evaluation tool.**
- **Configure your model architecture, build your dataset processing chain, define your objective function, and run your training, all with just a few clicks.** No code needed, just start drag some sliders and see how it changes the models behavior.
- **Real time progress and timing stats, inference visualization, and loss graphing** with a highly configurable training interface allowing you to put the most focus on whatever metrics you're most interested in seeing.
- **A testing interface allowing you to quickly load and validate your models on test datasets.** Define the epoch to load and how many images, and your model will be automatically run on a random selection of the provided images. The results will be evaluated with various loss functions and all the images and information will be displayed to you in the interface where you can sort by these metrics. Multiple tests can also be run and the results of each can be quickly swapped between directly from the UI.
- **Threaded training and testing framework** ensures that the UI remains responsive during training and that data visualization has a minimal effect on training and testing speed.
- **Training can easily be paused and resumed at any time.**
- **Load and visualize your model's training example images and loss log data** with the review panel. All [x, y, y_fake] example sets will be automatically loaded, and each loss in the log will be graphed with a configurable sample rate.
- **View time statistics during training** including slowest, fastest, and average epoch and iteration time, total train time, and a real time time graph of all previous epoch times.
### Easy Dataset Augmentation
- **A simple but powerful augmentation UI** allows you to quickly apply a variety of Albumentations-based augmentations to your datasets at training-time to help expand small datasets and improve model generalization.
- **Anything from random flipping and rotation, to optical distortion and random grayscale** can be applied by just dragging a slider.
### ONNX Conversion and Testing
- **The Toolbox UI also has a tool to convert your trained model to ONNX format.** There is also a developer friendly converter class which you can plug in to your custom pipeline to convert your models.
- **Also included is a tool to test your converted ONNX models.** It will take your model and a provided folder of test images, and it will run the model's inference on each image in the test directory, displaying the (A | B) results live in the UI.
### Dataset Tools
 **Also included with NectarGAN Toolbox are a set of helpful dataset processing tools. These currently include:** 
- **A tool to pair (A | B) images into Pix2pix input data**, extremely quickly, with control over direction and optional image scaling.
- **A tool to sort image files by various metrics** (white pixel count, black pixel count, mean pixel value, various types of contrast), and then two related tools. One to unsort the files back to their original order, and one to copy to sorting order of one directory of files to another. These can be used to help process and find bad images in large datasets.
- **A tool to automatically build train/test/val splits for dataset images** with easy control over split percentages.