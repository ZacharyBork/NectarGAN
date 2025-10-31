# NectarGAN Toolbox - Dataset
> [*`NectarGAN Toolbox - Home`*](../toolbox.md)
#### This section houses settings related to dataset loading and train-time data augmentations. 

> [!IMPORTANT]
> All data augmentations have a chance (defined by their percentage parameter) of being applied **any time a dataset image is loaded,** Not just the first time a given image is loaded. So a dataset image may be loaded with one set of augmentations one epoch, and a completely different set the next epoch.

### Dataset Files
| Setting | Description |
| :---: | --- |
| Dataset Root | The root directory for your dataset files. This directory should have subdirectories called `train`, `val`, and, optionally, `test`. These subdirectories should contain your paired (A | B) training images. |
| Load Size | The resolution (^2) to load **each** dataset image at. This is separate from `Crop Size`, which is the actual resolution of the images used during training, although they can be the same value if you don't want to use random cropping. |
| Crop Size | The resolution (^2) of the images used during training. If this is smaller than `Load Size`, an Albumentations RandomCrop operation will be performed to reach this target size. |
### Loading
| Setting | Description |
| :---: | --- |
| Direction | What direction to load the dataset (or, in simpler terms, which image is input and which is output where A is the left image, and B is the right image.) |
| Batch Size | The batch size to use during training. |
### Augmentations (Input)
| Setting | Description |
| :---: | --- |
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
| Compression Quality | Min and max quality for the applied compression operation. |
### Augmentations (Both)
| Setting | Description |
| :---: | --- |
| Horizontal Flip | The precentage chance to flip each image of any given dataset image along the horizontal axis. |
| Vertical Flip | The percentage chance to flip each image of any given dataset image along the vertical axis. |
| 90° Rotation | The percentage chance to apply a random 90° stepped rotation to each image of any given dataset pair. |
| Elastic Xform | The percentage chance to apply an elastic transformation to each image of any given dataset pair. |
| Elastic Alpha | Scaling factor for the elastic distortion. Higher values will produce more deformation. |
| Elastic Sigma | Smoothing factor for the distortions. Higher values will produce smoother results. |
| Optical Distort | Percentage chance to apply an optical distortion effect to any given dataset image. |
| Distortion Limits | Min and max range of the applied distortions. |
| Distortion Mode | What type of optical distortion to apply (e.g. camera or fisheye distortion). |
| Coarse Dropout | The percentage chance to apply coarse dropout (i.e. random rectangular cutouts) to any given dataset image. |
| Hole Count | The min and max number of holes to add if coarse dropout is applied to an image. |
| Hole Height | Min and max allowable height of each hole. |
| Hole Width | Min and max allowable width of each hole. |

---
*See Also:*
| [Toolbox - Home](../toolbox.md) | [Experiment](experiment.md) | [Training](training.md) | [Testing](testing.md) | [Review](review.md) | [Utilities](utilities.md) | [Settings](settings.md) |