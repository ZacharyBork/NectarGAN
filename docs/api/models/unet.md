# NectarGAN API - UNet
> [*`NectarGAN API - Home`*](/docs/api.md)
#### The NectarGAN API includes a modular, configurable UNet-style generator model.

## UNet Model
> **Reference**: [`nectargan.models.unet.model`](/nectargan/models/unet/model.py)

### Initializing a `UnetGenerator`
We'll start by having a look at the `UnetGenerator`. Looking at it's `__init__` function, we can see that it takes the following arguments:
| Arguments | Description |
| :---: | --- |
`input_size` | The intended input resolution (^2) for the generator. This is used for validation of training image resolution against downsampling layer count.
`in_channels` | The number of channels in the input dataset images (i.e. `1` for `mono`, `3` for `RGB`). **_Currently, 3 is the only known supported value._ Though technically, that's not actually true of the generator model. It should work with other values. The [`dataset loader`](/nectargan/dataset/paired_dataset.py) will not in its current state, however.**
`features` | The number of features on the first downsampling layer. Feature count for any given conv layer is capped at `features * 8`. 
`n_downs` | The number of downsampling layers (*Note: This value does not include the first layer or the bottleneck. This will be explained in greater detail below.*).
`use_dropout_layers` | The number of upsampling layers (starting from the deepest layer) to apply dropout on.
`block_type` | What block type to use when assembling the generator model.
`upconv_type` | What upsampling method to use:<br><br>- `Transposed` : Transposed convolution.<br>- `Bilinear` : Bilinear upsampling, then convolution.<br><br>Transposed convolution is the upsampling method tranditionally used in the Pix2pix model. However, bilinear upsampling + convolution can help to eliminate the checkboard artifacting which is commonly seen in these models. See [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190) for more info.

**So then, with these arguments in mind, a `UnetGenerator` can be instantiated as follows:** 
```python
from nectargan.models.unet.model import UnetGenerator
from nectargan.models.unet.blocks import UnetBlock

generator = UnetGenerator(
    input_size=256,          # Train on 256^2 resolution images
    in_channels=3,           # with 3 channels: (R, G, B)
    features=64,             # 64 output features on first down layer (with a cap of 512)
    n_downs=6,               # 6 downsampling layers (+1 for initial, +1 for bottleneck)
    block_type=UnetBlock,    # Or ResidualUnetBlock
    upconv_type='Transposed' # Or 'Bilinear'
)
```
### UNet Architecture
**Let's quickly go over how the UNet architecture actually works, so that we can better understand how it's implemented in the `UnetGenerator`.** This will be a relatively brief explanation, and it will have some prerequisite knowledge, so if you are totally unfamiliar with UNet, I first encourage you to check out the resources below. If you already have a good understanding of the UNet architecture, feel free to skip this section and move on to the NectarGAN implementation.

**Resources:**
- [Original UNet Paper](https://arxiv.org/pdf/1505.04597)
- Aladdin Persson's Videos on UNet: [1](https://www.youtube.com/watch?v=oLvmLJkmXuc), [2](https://www.youtube.com/watch?v=IHq1t7NxS8k), [3](https://www.youtube.com/watch?v=SuddDSqGRzg)
- [Towards Data Science | Understanding U-Net](https://towardsdatascience.com/understanding-u-net-61276b10f360/)

---
> ##### From [`nectargan.models.unet.model`](/nectargan/models/unet/model.py)
> ```python
>                 UNet-esque generator architecture
> ------------------------------------------------------------------------
> Input: [1, 3, 512, 512] (512^2 RGB)                     Output:  [1, 3, 512, 512]
>                         ↓                                 ↑
>   |Input Shape|    |Down Layer|    |Output/Skip Shape| |Up Layer|  |Output Shape|
>                         ↓                                 ↑
> [1, 3, 512, 512] ---> init_down -> [1, 64, 256, 256] --> final_up --> [1, 3, 512, 512]
>                         ↓                                 ↑
> [1, 64, 256, 256] --> down1 -----> [1, 128, 128, 128] -> up7 -------> [1, 64, 256, 256]
>                         ↓                                 ↑
> [1, 128, 128, 128] -> down2 -----> [1, 256, 64, 64] ---> up6 -------> [1, 128, 128, 128]
>                         ↓                                 ↑
> [1, 256, 64, 64] ---> down3 -----> [1, 512, 32, 32] ---> up5 -------> [1, 256, 64, 64]
>                         ↓                                 ↑
> [1, 512, 32, 32] ---> down4 -----> [1, 512, 16, 16] ---> up4 -------> [1, 512, 32, 32]
>                         ↓                                 ↑
> [1, 512, 16, 16] ---> down5 -----> [1, 512, 8, 8] -----> up3 -------> [1, 512, 16, 16]
>                         ↓                                 ↑ 
> [1, 512, 8, 8] -----> down6 -----> [1, 512, 4, 4] -----> up2 -------> [1, 512, 8, 8]
>                         ↓                                 ↑
> [1, 512, 4, 4] ---> bottleneck --> [1, 512, 2, 2] → → → → up1 -> [1, 512, 4, 4]
> ```
This is a diagram I made to help me conceptualize the tensor shapes of the inputs and skip connections at various depths in a UNet-style architecture (pardon the formatting, I wrote it directly inside of the python file). **This diagram shows a network with an input which has 3 `in_channels` (R, G, B), 64 `features`, and an input resolution of 512^2.**

**Let's walk through it, starting at the top left:**
> [!NOTE]
> **Skip connections:** *(This is explained in more depth below, but here is a quick explanation to start.)*
>
>  For every `down` layer, the output tensor is passed to the next down layer as input, but it is also passed via the skip connection to the corresponding `up` layer (**not directly, but concatenated with the output tensor of the up layer below**), as denoted by the rightward facing arrows. So, the output tensor from `init_down` is passed to the `final_up`, same for `down1` and `up7`, etc.
- We take our input (`[1, 3, 512, 512]`) and feed it into the `init_down` layer. This gives us an output tensor with the shape `[1, 64, 256, 256]`. So we halved the spatial resolution of the tensor, and increased the feature count to `64`, as defined by the generator's `features` value. 
- We take that output tensor (`[1, 64, 256, 256]`) and feed it in to `down1`, giving us an output shape of `[1, 128, 128, 128]`. Double the features, halve the resolution.
- We do this again for `down2` and `down3`. Now, though, we've hit our feature cap (`features * 8`, or `512`).
- So, for `down4`, `down5`, and `down6`, we continue to halve the spatial resolution, but the feature count remains at the cap of `512`
- Then we hit the bottleneck. As with the previous few downsampling layers, we halve the resolution, but keep the same feature count. This time, though, note that there is no skip connection, since there's obviously nothing to connect it to, as the output of the bottleneck is fed directly into the first upsampling layer, `up1`.

**Then, we have our upsampling path.** As noted, the first upsampling layer, `up1`, takes the output tensor of the bottneck layer directly. This layer doubles the spatial resolution of the tensor, but, mirroring the feature counts on the downsampling path, this layer keeps the feature count at the `512` cap. This results in an output tensor shape of `[1, 512, 4, 4]`.

**Then things start to become interesting.** Note that the output tensor from `down6` has the same shape as the output tensor from `up1`. This allows us to take those two tensors and stack the feature maps before passing them to the next up layer. These `skip connections` are the magic of the UNet architecture (and a couple others. [ResNet](https://arxiv.org/pdf/1512.03385), for example, also uses skip connections, albeit in a slightly different manner and for a slightly different reason). 

By stacking the feature map taken directly from the corresponding downsampling layer with the one from the previous upsampling layer, we allow the network to preserve sharp or fine details that would otherwise be lost with upsampling alone. 

A visual example of the concept:
![_skip_connection_visual_](/docs/resources/images/skip_connection_visual.png)
We can see that, were we to use just directly use the result of the upsampling path, we would lose a significant amount of detail. However, if we concatenate it with the output of the corresponding downsampling layer, we preserve a lot of that fine detail.

> [!NOTE]
> The above is just a visual example. This isn't exactly what's happening, but the concept and result are very similar.

### Channel Mapping

## UNet Blocks
> **Reference**: [`nectargan.models.unet.blocks`](/nectargan/models/unet/blocks.py)

## References
- [*U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)*](https://arxiv.org/pdf/1505.04597)
- [*Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) (Md Zahangir Alom et al.)*](https://arxiv.org/pdf/1802.06955)
- [*ResUNet: A Deep Neural Network for Semantic Segmentation of Urban Scenes (Diakogiannis et al.)*](https://arxiv.org/pdf/1904.00592)
- [*Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks (Isola et al., 2017)*](https://arxiv.org/abs/1611.07004)