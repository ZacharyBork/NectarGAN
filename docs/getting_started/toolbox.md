# NectarGAN - Getting Started (Toolbox)
#### A graphical tool for training and testing models, reviewing results of previous tests, converting models to ONNX and testing the resulting model, and processing dataset files, all packaged into a single modern and easy to use interface..
> [!NOTE]
> For this walkthrough, we will be using the ubiquitous `facades` Pix2pix dataset, kindly provided by the University of California, Berkeley. Before beginning, please head to the following link and select `facades.tar.gz` to download this dataset if you would like to follow along:
>
> https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/

##### Reference: [`NectarGAN Toolbox Documentation`](/docs/toolbox.md)

## Launching the ToolBox
With NectarGAN installed via your preferred method, you can launch the Nectargan Toolbox as follows:
```bash
python -m nectargan.start.toolbox
```
The Toolbox will open to a screen which looks like this:
![_toolbox_experiment_](/docs/resources/images/toolbox/toolbox_experiment.PNG)

## Experiment Settings
**On this screen, we can define output settings for our current experiment, and also define architecture options for our generator and discriminator, if we so choose.**

**For now, we will:**
1. Select an output root, either by pasting the path into the `Output Root` or by using the `Browse` button and selecting a directory with the file explorer. The output root is, as the name suggests, the root output directory, wherein a unique subdirectory will created for each experiment which is run.
2. Set an `Experiment Name`. This will be used to name the experiment output subdirectory. For now, we will set it to `facades`. 
3. As we have no other versions yet, we will leave `Version` at `1`, though it should be noted that `Version` is always kind of optional, as experiment outputs will be versioned automatically regardless.

## Dataset Settings
**Then, we will move on to `Dataset` options by clicking the blue `Dataset` button on the left hand bar.** After doing so, you will be presented with a screen that looks like this:
![_toolbox_dataset_](/docs/resources/images/toolbox/toolbox_dataset.PNG)
**Here, you can control settings related to your dataset, including load and crop size, load direction, and a full suite of built in augmentation options.** Clicking on the check box next to the augmentation type will expand the augmentation options for that type. The names are fairly self-explanatory, but just to be sure:
- `Input` : Augmentations applied only to the input image (`A` in `AtoB`, `B` in `BtoA`)
- &nbsp;`Both`&nbsp; : Augmentation applied to both the input and the target image.

**For now, on this screen, we will:**
1. Unzip the `facades` dataset (twice), if we haven't already.
2. Set the `Dataset Root` to the path to the `facades` dataset folder.
> [!IMPORTANT]
> The path we input for `Dataset Root` should be the root directory for the dataset, i.e. the directory which contains the `train`, `test`, and `val` subdirectories.
3. If we look at one of our dataset images, we can see it has a resolution of `[512x256]`, meaning each image (i.e. `input`, `target`) has a resolution of `[256x256]`. As such, we do not need to change our `Load Size` or `Crop Size` here, as the default values are set to both load and random crop at `256^2` which, with this dataset, will just load images at full resolution and apply no cropping.
4. We will change the `Direction` to `BtoA`. Looking again at our dataset images, we can see that the image on the right, the `B` image, is the semantic mask. We want to turn that in to the image on the left (`A`), the building facade image. If you'd rather see the model try to learn the semantic segmentation task instead though, feel free to leave it at `AtoB`.

## Training Settings
**And finally, we're ready to move on to the training settings, which can be done by clicking on the purple `Training` button on the left-hand bar.** After doing so, you will be greeted with a window which looks like this:
![_toolbox_training_](/docs/resources/images/toolbox/toolbox_training.PNG)
**Here, you can set various options related to learning rate schedules for the generator and discriminator, settings related to loss weighting and logging, and also some options related to checkpoint and example image saving during training.**

Also of note is the `Continue Train` checkbox. Clicking this will present you with the option to select an epoch to load, and then when you click train, the checkpoint file for that epoch will be loaded to continue model training.

**For now, we don't need to change anything here.** The default settings will initialize the learning rate schedule and loss weights as they were outlined in the original Pix2pix paper[^1], but you are encouraged to come back after this walkthrough to start playing around with the settings here, they can change the behaviour of the generator in lots of fun and interesting ways depending on the task.

## Start Training
> ##### Toolbox training on the `facades` dataset *(epoch 173)*.
> ![_toolbox_running_](/docs/resources/gifs/toolbox_training.gif)


***So then with all that done, we are ready to start training!*** This can be done by clicking the `Begin Training` training button. When we do, a few things will happen:
1. The UI will lock. Most settings can't be changed during training with the exception of some in the main settings panel (accessed via the red `Settings` button on the left-hand bar.)
2. A new subdirectory will be created in the `Output Root` named `{experiment_name}_v{version}`.
3. In that subdirectory, a few things will be created: a `loss_log.json`, a `train1_config.json`, and a subdirectory called  `examples` for exporting example images to during training.
4. We will be presented with two new buttons, `Pause Training`, to pause (and continue) training at any point, and `Stop Training`, which will stop training altogether and return the UI to its previous state.
5. After just a second, we will start seeing example output images, and losses will start being graphed. The rate at which these updates happen can be changed (even during training) in the main `Settings` panel.
6. After each epoch, the model will be loaded automatically in eval mode and run on an image from the `val` set, and an example image set will be exported to the `examples` directory in the experiment directory.
7. After a few epochs (5 if you have followed this guide exactly up until this point), cached loss data will be exported to the loss log.

**And that's really all there is to training with the NectarGAN Toolbox.** Let's let the model train for a bit (maybe 10 or 20 epochs, shown by the epoch counter in the bottom right), then we'll come back once we have some checkpoint files and example images to play with.
## Model Testing