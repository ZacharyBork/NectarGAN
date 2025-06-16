# NectarGAN Toolbox
## What is it?
The NectarGAN Toolbox is a graphical tool for training and testing models, reviewing results of previous tests, converting models to ONNX and testing the resulting model, and processing dataset files.

It can be launched by running `python -m scripts.toolbox` from the repository root.

>The Toolbox interface is broken down into seven core sections, each of which can be accessed via the buttons of the left-hand bar, or by pressing `Ctrl+[1-7]`.
## Experiment
Here you will initialize output settings and architectural settings for the generator and discriminator.
| Setting | Description |
| --- | --- |
| Output Root | Root output directory. A new experiment directory will be created in this directory each time you hit train. |
| Experiment Name | The name of the experiment. This will be used to name the experiment output directory. |
| Version | The version number to append to the experiment number. When conducting multiple experiments with the same name, this parameter is optional. Experiment output directories will be versioned automatically. |
| Features | Number of output features on the first downsampling layer. The generator model has a feature cap equal to this value * 8. |
| Down Layers | Number of downsampling layers (and matching upsampling layers) to add when assembling the generator. |
| Block Type | What block type to use for the generator (i.e. UNet block, Residual UNet block). |
| Upsampling Type | What type of upsampling to perform in the generator (i.e. TransposedConv2D or bilinear upsampling + Conv2D). Transposed convolution is oftentimes responsible for the "checkerboard" artifacting that is notorious in pixel-to-pixel GANs. See [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190) for more info. |
| Layer Count | Number of downsampling layers to add to the discriminator. |
| Base Channels | Number of output channels on the first discriminator layer. |
| Max Channels | Maximum allowed channel count for any given layer in the disriminator. |
## Dataset
This section houses settings related to dataset loading and train-time data augmentations. All data augmentations have a chance (defined by their percentage parameter) of being applied **any time a dataset image is loaded,** Not just the first time a given image is loaded. So a dataset image may be loaded with one set of augmentations one epoch, and a completely different set the next epoch.
### Dataset Files
| Setting | Description |
| --- | --- |
| Dataset Root | The root directory for your dataset files. This directory should have subdirectories called `train`, `val`, and, optionally, `test`. These subdirectories should contain your paired (A | B) training images. |
| Load Size | The resolution (^2) to load **each** dataset image at. This is separate from `Crop Size`, which is the actual resolution of the images used during training, although they can be the same value if you don't want to use random cropping. |
| Crop Size | The resolution (^2) of the images used during training. If this is smaller than `Load Size`, an Albumentations RandomCrop operation will be performed to reach this target size. |
### Loading
| Setting | Description |
| --- | --- |
| Direction | What direction to load the dataset (or, in simpler terms, which image is input and which is output where A is the left image, and B is the right image.) |
| Batch Size | The batch size to use during training. |
### Augmentations (Input)
| Setting | Description |
| --- | --- |
| Colorjitter | The percentage chance to apply random colorjitter to any given input image. |
| Colorjitter Range | The min and max allowed values of the colorjitter if it is applied. |
| Gaussian Noise | The percentage chance to apply gaussian noise to any given input image. |
| Gaussian Range | The min and max range of the gaussian noise. |
| Motion Blur | The percentage chance to apply a pseudo-motion blur effect to any given input image. |
| Motion Blur Limit | The maximum kernal size for the blur operation. The minimum allowed kernal size is 3 so this value should be >3. |
| Random Gamma | The percentage chance to randomize the gamma of any given input image. |
| Gamma Limits | The upper and lower bounds (in %) of the random gamma operation. |
| Grayscale | The percentage chance to convert any image to grayscale |
| Grayscale Method | What method to use when converting dataset images to grayscale. |
| Compression | The percentage chance to apply simulated compression artifacts to any given dataset image. |
| Compression Type | What type of compression artifacts to apply (e.g. JPEG, WEBP). |
| Compression Quality | Min and max quality for the applied commpression operation. |
### Augmentations (Both)
| Setting | Description |
| --- | --- |
| Horizontal Flip | The precentage chance to flip each image of any given dataset image along the horizontal axis. |
| Vertical Flip | The percentage chance to flip each image of any given dataset image along the vertical axis. |
| 90° Rotation | The percentage chance to apply a random 90° stepped rotation to each image of any given dataset pair. |
| Elastic Xform | The percentage chance to apply an elastic transformation to each image of any given dataset pair. |
| Elastic Alpha | Scaling factor for the elastic distortion. Higher values will produce more deformation. |
| Elastic Sigma | Smoothing factor for the distortions. Higher values will produce smoother results. |
| Optical Distort | Percentage chance to apply an optical distorion effect to any given dataset image. |
| Distortion Limits | Min and max range of the applied distortions. |
| Distortion Mode | What type of optical distortion to apply (e.g. camera or fisheye distortion). |
| Coarse Dropout | The percentage chance to apply coarse dropout (i.e. random rectangular cutouts) to any given dataset image. |
| Hole Count | The min and max number of holes to add if coarse dropout is applied to an image. |
| Hole Height | Min and max allowable height of each hole. |
| Hole Width | Min and max allowable width of each hole. |
## Train
Here you will find settings related to model training, loss weighting and logging, and example and checkpoint saving.
| Setting | Description |
| --- | --- |
| Continue Train | If checked, training will be continued from a previous set of checkpoint files. |
| Load Epoch | What epoch checkpoints to load when continuing training from previously trained weights. |
### Generator
| Setting | Description |
| --- | --- |
| Epochs | How many epochs to run at the `Initial` learning rate. |
| Epochs Decay | Over how many epochs to decay from the `Initial` learning rate, down to the `Target` learning rate. |
| Initial | The initial learning rate to use when training the model. The model will be trained at this learning rate for a number of epochs defined by `Epochs`. |
| Target | The learning rate which, after the initial epochs are completed, the scheduler will decay the generator's learning rate to over a number of epochs defined by `Epochs Decay`. |
| Beta1 | The momentum term for the optimizer (Adam in this case). Higher values will allow the optimizer to more rapidly respond to bad habits learned by the generator during training, but can cause also cause instability. |
### Discriminator
| Setting | Description |
| --- | --- |
| Separate Learning Schedules | If checked, the discriminator can be assigned a different learning rate schedule than the one used for the generator. |
| Epochs | See Generator options. |
| Epochs Decay | See Generator options. |
| Initial | See Generator options. |
| Target | See Generator options. |
| Beta1 | See Generator options. |
### Loss
| Setting | Description |
| --- | --- |
| GAN | The weight value to apply to the generators adversarial loss. Higher values will cause the generator to try harder to fool the discriminator, at the cost of higher deviation from the ground truth. |
| L1 | The weight value to apply to the generators L1 Loss. Higher values will penalize the generator more harshly for pixel-wise devations from the ground truth. |
| L2 | MSELoss, basically `L1^2`. The behaves very similarly to L2 except the bigger the pixel-wise error, the more, relatively speaking, the generator is punished. |
| Sobel | Sobel structure loss. Derives a loss value for the generator by applying a Sobel operation to the generated fake and the ground truth, and comparing the two results with a pixel-wise loss function (L1 in this implementation). Higher values can cause the generator to better preserve large scale structural detail. |
| Laplacian | Laplacian structure loss. Similar to Sobel loss but with a Laplacian kernel instead of a Sobel kernel. Higher values can cause the generator to preserve more fine, textural detail from ground truths. |
| VGG | Perceptual loss with VGG19. Takes the generated image and the ground truth, feeds each to the pre-trained VGG19 image classification model and extracts feature maps at various depths, then compares the feature maps for each with L1 loss. This creates a cost function that encourages the generator to create images that are visually similar to the ground truth images, while punishing it far less harshly for small, pixel-level deviations. This can sometimes encourage the generator create images with extremely high visual realism, and also help reduce the blurring (or averaging) you sometimes see with traditional L1 loss in pixel to pixel models. |
| Log Losses During Training | If enables, loss data will be cached during training and periodically dumped to the loss log in the experiment directory. |
| Log Dump Frequency (epochs) | The frequency, in epochs, to dump the cached loss values to the log file. Realistically, unless you are working on very large datasets with relatively high epoch counts, it is safe to have the value fairly high. |
## Test
Here you will find settings related to testing previously trained models.
| Setting | Description |
| --- | --- |
| Experiment | The system path to the experiment directory for the experiment you would like to test. |
| Load Epoch | The checkpoint epoch you would like to load for testing. |
| Test Iterations | The number of images from the test dataset to run the model on. |
| Override Dataset | If enabled, this will allow you to input the path to a dataset to use from testing, rather than using the one defined on the `Dataset` tab. |
| Override Dataset Path | If override dataset is enabled, this path will be used for the test dataset rather than the one in the `Dataset` tab. |
## Review

## Utilities

## Settings