# NectarGAN Toolbox - Experiment
#### Here you will initialize output settings and architectural settings for the generator and discriminator.

| Setting | Description |
| :---: | --- |
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

---
*See Also:*
| [Toolbox - Home](/docs/toolbox.md) | [Dataset](/docs/toolbox/dataset.md) | [Training](/docs/toolbox/training.md) | [Testing](/docs/toolbox/testing.md) | [Review](/docs/toolbox/review.md) | [Utilities](/docs/toolbox/utilities.md) | [Settings](/docs/toolbox/settings.md) |