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
`block_type` | What block type to use when assembling the generator model (see [here](/docs/api/models/unet_blocks.md) for more info).
`upconv_type` | What upsampling method to use:<br><br>- `Transposed` : Transposed convolution.<br>- `Bilinear` : Bilinear upsampling, then convolution.<br><br>Transposed convolution is the upsampling method traditionally used in the Pix2pix model. However, bilinear upsampling + convolution can help to eliminate the checkboard artifacting which is commonly seen in these models. See [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190) for more info.

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
>  For every `down` layer, the output tensor is passed to the next down layer as input, but it is also passed via the skip connection to the corresponding `up` layer, as denoted by the rightward facing arrows (**not directly passed as input, but concatenated with the output tensor of the up layer below**). So, the output tensor from `init_down` is passed to the `final_up`, same for `down1` and `up7`, etc.
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
> **Reference:** [`nectargan.models.unet.model.UnetGenerator.build_channel_map()`](/nectargan/models/unet/model.py#L94)

Alright, now that we've covered the basics of how the UNet architecture functions, let's now have a look at how it's implemented in NectarGAN. The core of this is the channel mapping function. This function assembles two lists of tuples, one for the downsampling channels, and another for the upsampling channels. Each tuple in each list corresponds to a layer, and contains two `int` values:

1. The number of input channels for the layer.
2. The number of output channels for the layer.

Let's walk through the function step by step to understand how those lists are created:
1. We create the first of our two lists, `down_channels`. In doing so, we also add the first tuple to it with the input and output channel counts for the first downsampling layer, since we know this is always going to be (`in_channels`, `features`).
2. We create two variables, `in_features` and `out_features`, and initialize both to the `features` value used when initializing the `UnetGenerator`.
3. We create a loop with an iteration count of `n_downs`, in which we:
    - First, set `in_features` to the max of `out_features`, and `features*8`, because we want to make sure it doesn't go beyond that hard cap.
    - Then, we set `out_features` to the min of `out_features*2` and `features*8`. **This defines the doubling of the feature maps with each encoder layer.**
    - We append a new entry to `down_channels` containing our new `in_channels` and `out_channels`. Then do it again for the next layer, and the next, and so on. So the `out_channels` value of the previous iteration becomes the `in_channels` value of the next iteration. Up to the point where they cap out at `features*8`. The best way to illustrate this is just to list the results for each iteration, like so:
    ```json
    NOTE: This example assumes an `input_channels` value of `3` and a `features` value of 64.

    First Layer: (3, 64)
    Iteration 1: (64, 128)
    Iteration 2: (128,256)
    Iteration 3: (256, 512)
    Iteration 4: (512, 512)

    And now we've reached our maximum feature count, all subsequent layers will have a value of (512, 512).
    ```
4. Now, we create a list for our skip connection channel sizes. To do this, we perform two operations at once:
    - We reverse the list of `down_channels`.
    - While doing so, we also flip the `in_channels` and `out_channels` of each entry, since the direction is reversed on the upsampling path and the channel sizes need to reflect that.
5. Now, we create our other list, `up_channels`, which holds the channel sizes for the layers in the upsampling path. To do this, we just make a new list from our `skip_channels` list, but for each entry, we double the `in_channels` value, to reflect the fact that we are going to be concatenting the skip tensor with the previous upsampling tensor for the input of each of these layers.
6. Finally, we insert one additional layer at the beginning of `up_channels`. This represents the deepest upsampling layer, which takes its input directly from the bottleneck layer.

Then at the very end, we make a dictionary from the lists to make it a bit easier to look up our values when defining our actual layers.

### Layer Definitions
The layers in the `UnetGenerator` are defined by these three functions:

1. [`define_downsampling_blocks`](/nectargan/models/unet/model.py#L121)
2. [`define_bottleneck`](/nectargan/models/unet/model.py#L139)
3. [`define_upsampling_blocks`](/nectargan/models/unet/model.py#L148)

We won't go too deep in to these, this document is getting long enough as it is. But in short, these three functions use the channel map which we discussed above to define the blocks in the upsampling, downsampling, and bottleneck layers, setting the correct values based on layer type and depth. The default configuration sets the values exactly as a traditional UNet does.

### Forward
> **Reference:** [`nectargan.models.unet.model.UnetGenerator.forward()`](/nectargan/models/unet/model.py#L180)

Lastly, we will have a look at the `forward` function for the `UnetGenerator` class. This function uses the layers defined by the above three functions to assemble a UNet style architecture and passes the input tensor `x` through it.

**Walking through this function, we:**
1. Run the tensor through the initial downsampling layer, halving the spatial resolution, and taking the feature count from the value of `in_channels` to the value of `features`.
2. Create a new list, `skips`, to store our skip connection tensors, and add the result of the initial downsampling layer to it.
3. We loop through all of our additional downsampling layers, running the tensor through each subsequent one and appending the resulting tensor to our list of `skips`.
4. Reverse our list of skips to align them with the input tensors in the upsampling path.
5. Run our `x` tensor through the bottleneck layer. Note that we do not append the result to `skips`, since the bottleneck just passes its results directly to the first upsampling layer.
6. Loop through all of our sampling layers. For the first layer, we just run it through the layer directly, for the reason just noted. Then for each layer after that, we first grab the corresponding skip connection, then we concatenate the `x` tensor with the correspondinding skip tensor along the channel dimension. Then, finally, we input that concatenated tensor into the upsampling layer.
7. After we finish the loop of upsampling layers, we take the resulting `x` tensor, concatenate it with the final skip tensor, and feed it in to the final upsampling layer, returning the result.

**And that's basically all there is to the `UnetGenerator`. We've now seen how the channel map and layers are assembled when the generator is initialized, and we've seen how that architecture is used whenever the generator's `forward` function is called. The last note to touch on is...**

## UNet Blocks
> **Reference**: [`nectargan.models.unet.blocks`](/nectargan/models/unet/blocks.py)

Please see [here](/docs/api/models/unet_blocks.md) for more information about the drop-in block system used by the `UnetGenerator`.

## References
- [*U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)*](https://arxiv.org/pdf/1505.04597)
- [*Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) (Md Zahangir Alom et al.)*](https://arxiv.org/pdf/1802.06955)
- [*ResUNet: A Deep Neural Network for Semantic Segmentation of Urban Scenes (Diakogiannis et al.)*](https://arxiv.org/pdf/1904.00592)
- [*Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks (Isola et al., 2017)*](https://arxiv.org/abs/1611.07004)