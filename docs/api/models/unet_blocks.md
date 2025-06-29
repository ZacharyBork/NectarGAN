# NectarGAN API - UNet Blocks
> [*`NectarGAN API - Home`*](/docs/api.md)
#### The NectarGAN UNet generator accepts drop-in convolutional blocks which which can dramatically change the behaviour of the model. Currently, there are two built in block types.

> [!NOTE]
> This document is intended as a followup of the [UNet API documentation](/docs/api/models/unet.md). It is advisable to read that document before beginning this one.

## What is a UNet Block?
In the simplest terms, a Unet block in NectarGAN is just a class which inherits from torch.nn.Module, and which has a forward function that takes a tensor as input, and which returns a tensor. However, there are a few additional requirements.

1. **Each UNet block class in NectarGAN is responsible for _both upsampling and downsampling_.**
2. Currently all UNet blocks in NectarGAN must take a set list of arguments, which are set by the UNet generator based on the layer type and depth. **They don't need to use every argument**, but they must accept them (*This may change in the future though. This was a fairly early API component and could do with an overhaul.*).<br><br>**These arguments are:**
    | Argument | Type | Description |
    | :---: | :---: | --- |
    `in_channels` | (`int`) | The input channel count for the layer. Defined by the layer type and depth based on the result of the `UnetGenerator` channel mapping function.
    `out_channels` | (`int`) | The output channel count for the layer. Defined by the layer type and depth based on the result of the `UnetGenerator` channel mapping function.
    `upconv_type` | (`str`) | What upsampling method to use:<br><br>- `Transposed` : Transposed convolution.<br>- `Bilinear` : Bilinear upsampling, then convolution.<br><br>Transposed convolution is the upsampling method traditionally used in the Pix2pix model. However, bilinear upsampling + convolution can help to eliminate the checkboard artifacting which is commonly seen in these models. See [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190) for more info.
    `activation` | (`str\|None`) | The activation type for the layer. There are four values which are passed to the blocks by the generator, and which should be processed accordingly by the block. These are:<br><br>- `'leaky'` : `torch.nn.LeakyReLu()`<br>- `'relu'` : `torch.nn.ReLU()`<br>- `'tanh'` : `torch.nn.Tanh()`<br>- `None` : `pass`<br><br>*Technically, these can be processed however you want. The generator model passes each at the stages they are supposed to be used in the UNet architecture though,* **so using modules which behave differently will likely cause strange behaviour.**
    `norm` | (`str\|None`) | What type of normalization to use. Currently, there are only two values which will ever be passed by the UNet generator model:<br><br>- `instance` : `nn.InstanceNorm2d(out_channels)`<br>- `None` : `pass`<br><br>*Again, these can techincally be whatever, but these values are what the model is expecting it to be.* **This will also likely change in the future once normalization methods get plugged in to the Toolbox UI.**
    `down` | (`bool`) | On downsampling layers (including the bottleneck layer), the `UnetGenerator` will pass a value of `True` for this argument. On upsampling layers, it will pass a value of `False`. This should be used to determine whether to run convolution or transposed convolution (or bilinear + conv2d) when the block is called.
    `bias` | (`bool`) | The `UNetGenerator` will pass a value of `True` for this argument on the first downsampling layer, the bottleneck, and the last upsampling layer. On every other layer, it will be passed a value of `False`. Used as a direct passthrough for the bias argument in `torch.nn.Conv2d` and `torch.nn.ConvTranspose2d` in the base `UnetBlock`.
    `use_dropout` | (`bool`) | **This will be passed a value of `False` for every layer except the first X upsampling layers (starting from the deepest)**, with X defined by the `use_dropout_layers` argument used to initialize the generator (default `3`). For these layers, it will be passed a value of `True`. Used in the Unet blocks to define whether dropout should be applied to a given layer.
    
**Now then, let's have a quick look at the two block types currently included in the NectarGAN API.**
## UnetBlock
> **Reference:** [`nectargan.models.unet.blocks.UnetBlock`](/nectargan/models/unet/blocks.py#L7)
### `UnetBlock.__init__()`
This is a very standard UNet block. Walking through it, we:
1. Super init the `torch.nn.Module` parent class.
2. Create an empty list to append our modules to.
3. Check whether this is a downsampling block:
    - **If yes:** We just append a `Conv2d` module with `kernel_size=4`, `stride=2`, and `padding=1`. We also pass it the input `bias` and set its `padding_mode` to `reflect`.
    - **If No:** We do another check on the `upconv_type`. There are only two values it could be:
        1. `Transposed` : We just append a `torch.nn.ConvTranspose2d`, again with `kernel_size=4`, `stride=2`, and `padding=1`, and pass it the `bias` argument.
        2. `Bilinear` : Here we instead first append a `torch.nn.Upsample` with a `scale_factor` of `2` (double the spatial resolution). Then we apply reflection padding and use use a standard `Conv2d`, this time with a `stride` of `1` though, since we've already increased the resolution.
4. Check the value of the `norm` argument. If it's `instance`, we apply a `torch.nn.InstanceNorm2d`. If it's `None`, we do nothing.
5. Check the value of the `activation` argument, and apply the corresponding activation function, or pass if `None` (see the table above for more info).
6. Unpack the modules list into a `torch.nn.Sequential` and store it in `self.conv`.
7. Finally, store the value of the `use_dropout` argument and set `self.dropout` to `torch.nn.Dropout` with a probability of 50%.

**And that's it for setup, let's have a quick look at the `forward` function to see how we're using the modules we just assembled.**

### `UnetBlock.forward()`
The forward function for the `UnetBlock` class is super simple. 

It takes a `torch.Tensor`, `x`, as input, generally intended to be an input image from a training dataset as a tensor. All we do is run the `torch.nn.Sequential` we created in the `__init__` method, `self.conv`, and feed it the input tensor `x`. Then assign the output back to `x`. 

Then we check if we are using dropout for the current layer and, if so, apply dropout to the `x` tensor, then return it. Otherwise we just return it directly.

## ResidualUnetBlock
> **Reference:** [`nectargan.models.unet.blocks.ResidualUnetBlock`](/nectargan/models/unet/blocks.py#L65)

The `ResidualUnetBlock` is very similar to the `UnetBlock`. In fact, it actually inherits from the `UnetBlock` class, and just uses its `self.conv` directly for upsampling and downsampling. The only difference is that this block also includes a [residual connection](https://en.wikipedia.org/wiki/Residual_neural_network).

**You can think of these like the skip connections we discussed in the [previous document](/docs/api/models/unet.md),** only "shorter". Where the skip connections in the UNet architecture skip over all layers deeper than themselves, these residual connections skip only over the current layer's convolution or transposed convolution operation. They instead apply the same spatial transform operation, but with a `1x1` kernel, to match the main path's tensor shape, or just return a `torch.nn.Identity` in cases where the input and output channel counts are the same. This is often the case at the bottleneck, and is always the case once the generator's feature cap has been reached.

**Residual connections are used for different things in different architectures.** [ResNet](https://arxiv.org/pdf/1512.03385), the origin of residual connections in deep learning, primarily uses them to stabilize training, and to help avoid the [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem). ResNets are ***extremely*** deep networks, with popular variants having [34](https://huggingface.co/microsoft/resnet-34) and [50](https://huggingface.co/microsoft/resnet-50) layers, and with some even having upwards of [150](https://huggingface.co/microsoft/resnet-152). This presents a challenge, as the large number of sequential activation functions causes the backpropagation gradients to shrink to effectively nothing. This, in turn, causes the shallower layers in the network to learn at a slower and slower rate, until they eventually stop learning altogether. The residual connections in ResNet provide an alternate route for gradients to flow which avoids this shrinkage.

**Here though, their task is more simple.** They basically are serving exactly as the skip connections in the UNet are, except at a more "local" scale. The intent is the same though: to preserve sharp or fine details which might otherwise be lost during the encoding/decoding process.

---