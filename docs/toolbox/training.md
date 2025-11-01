# NectarGAN Toolbox - Training
> [*`NectarGAN Toolbox - Home`*](../toolbox.md)
#### Here you will find settings related to model training, loss weighting and logging, and example and checkpoint saving.

### Continue Training
| Setting | Description |
| :---: | --- |
| Continue Train | If checked, training will be continued from a previous set of checkpoint files. |
| Load Epoch | What epoch checkpoints to load when continuing training from previously trained weights. |
### Generator
| Setting | Description |
| :---: | --- |
| Epochs | How many epochs to run at the `Initial` learning rate. |
| Epochs Decay | Over how many epochs to decay from the `Initial` learning rate, down to the `Target` learning rate. |
| Initial | The initial learning rate to use when training the model. The model will be trained at this learning rate for a number of epochs defined by `Epochs`. |
| Target | The learning rate which, after the initial epochs are completed, the scheduler will decay the generator's learning rate to over a number of epochs defined by `Epochs Decay`. |
| Beta1 | The momentum term for the optimizer (Adam in this case). Higher values will allow the optimizer to more rapidly respond to bad habits learned by the generator during training, but can cause also cause instability. |
### Discriminator
| Setting | Description |
| :---: | --- |
| Separate Learning Schedules | If checked, the discriminator can be assigned a different learning rate schedule than the one used for the generator. |
| Epochs | See Generator options. |
| Epochs Decay | See Generator options. |
| Initial | See Generator options. |
| Target | See Generator options. |
| Beta1 | See Generator options. |
### Loss
| Setting | Description |
| :---: | --- |
| GAN | The weight value to apply to the generators adversarial loss. Higher values will cause the generator to try harder to fool the discriminator, at the cost of higher deviation from the ground truth. |
| L1 | The weight value to apply to the generators L1 Loss. Higher values will penalize the generator more harshly for pixel-wise devations from the ground truth. |
| L2 | MSELoss, basically `L1^2`. The behaves very similarly to L1 except the bigger the pixel-wise error, the more, relatively speaking, the generator is punished. |
| Sobel | Sobel structure loss. Derives a loss value for the generator by applying a Sobel operation to the generated fake and the ground truth, and comparing the two results with a pixel-wise loss function (L1 in this implementation). Higher values can cause the generator to better preserve large scale structural detail. |
| Laplacian | Laplacian structure loss. Similar to Sobel loss but with a Laplacian kernel instead of a Sobel kernel. Higher values can cause the generator to preserve more fine, textural detail from ground truths. |
| VGG | Perceptual loss with VGG19. Takes the generated image and the ground truth, feeds each to the pre-trained VGG19 image classification model and extracts feature maps at various depths, then compares the feature maps for each with L1 loss. This creates a cost function that encourages the generator to create images that are visually similar to the ground truth images, while punishing it far less harshly for small, pixel-level deviations. This can sometimes encourage the generator create images with extremely high visual realism, and also help reduce the blurring (or averaging) you sometimes see with traditional L1 loss in pixel to pixel models. |
| Log Losses During Training | If enables, loss data will be cached during training and periodically dumped to the loss log in the experiment directory. |
| Log Dump Frequency (epochs) | The frequency, in epochs, to dump the cached loss values to the log file. Realistically, unless you are working on very large datasets with relatively high epoch counts, it is safe to have the value fairly high. |

---
*See Also:*
| [Toolbox - Home](../toolbox.md) | [Experiment](experiment.md) | [Dataset](dataset.md) | [Testing](testing.md) | [Review](review.md) | [Utilities](utilities.md) | [Settings](settings.md) |