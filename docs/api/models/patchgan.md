# NectarGAN API - PatchGAN
> [*`NectarGAN API - Home`*](/docs/api.md)
#### NectarGAN includes a modular, configurable PatchGAN-style discriminator model.

## What is a PatchGAN?
**A PatchGAN is essentially just a convolutional neural network.** Where a traditional GAN disciminator reduces input to a single scalar value (real or fake), a PatchGAN instead convolves the input to an array of values, each of which represents whether the corresponding patch (a 70x70 pixel square by default) in the image is real or fake.

**I strongly encourage you to read this [article](https://sahiltinky94.medium.com/understanding-patchgan-9f3c8380c207).** The author provides a fantastic visual walkthrough of the inner workings of the architecture.
## PatchGAN Model
> **Reference:** [`nectargan.models.patchgan.model`](/nectargan/models/patchgan/model.py)

The root of the PatchGAN implementation in NectarGAN is the `Discriminator` class. This class assembles a PatchGAN style discriminator and provides a `forward` function for making predictions using the PatchGAN model.

### `Discriminator.__init__()`
Let's first look at the `Discriminator` class's [`__init__`](/nectargan/models/patchgan/model.py#L9) function. We can see it only takes 4 arguments:
| Argument | Type | Description |
| :---: | :---: | --- |
`in_channels` | `int` | The number of channels in the input images (i.e. `1` for mono, `3` for RGB).
`base_channels` | `int` | The number of output channels on the first conv layer.
`n_layers` | `int` | The number of conv layers to add to the discriminator model. The actual layer count of the discriminator will be `n_layers+1` because a final layer is added which reduces the channel count to `1` for the final predictions on each patch.
`max_channels` | `int` | The maximum allowed channels for any given conv layer in the model.

**Now let's quickly walk through this `__init__` function to see how it is used to assemble the network:**
1. We super init the parent `torch.nn.Module` class.
2. We create a new list, `self.layers`, to store our convolutional layers.
3. We run the function [`add_initial_layer()`](/nectargan/models/patchgan/model.py#L33). This function:
    - Appends a `torch.nn.Conv2d` module, setting the in channels to `in_channels * 2` (because remember, this network is passed two tensors, an input and ground truth, or a fake and ground truth), and set the output channels to `base_channels`. This layer uses a `4x4` kernel, `stride=2`, and reflection padding of `1`.
    - Appends a `torch.nn.LeakyReLU` for activation. 
4. We initialize a member variable, `self.in_ch`, to keep track of the current channel counts in the next stage, and set its value to the value of the `base_channels` argument.
5. We then run the function [`add_n_layers()`](/nectargan/models/patchgan/model.py#L33). This function adds `n_layers-1` conv layers to the discriminator. In does this in a loop, where each iteration:
    - Creates a variable, `out_channels`, and sets it to the min of `self.in_ch * 2`, and `max_channels`, doubling the channel count while enforcing the hard channel cap.
    - Picks a stride value for the current iteration's conv layer. This value is `2` for every layer except the final one, which instead uses a stride of `1`.
    - Create a `CNNBlock` module (explained below), setting the input channels to `self.in_ch` and the output channels to our new `out_ch` value, and passing it the `stride` value we just discussed. This `CNNBlock` module also uses a `4x4` kernel with reflection padding of `1`. However, it also includes a `torch.nn.InstanceNorm2d` and a `torch.nn.LeakyReLU` module.
    - Then we update `self.in_ch` with the new `out_ch` value for the next iteration.
6. After that, we add our final conv layer. This layer uses a `4x4` kernel with `stride=1` and `padding=1`. With a 256x256 input resolution, and with the layers we've assembled up to this point, this kernel size corresponds to a 70x70 pixel receptive field on the input image (hence the common name `70×70 PatchGAN`). This layer also reduces the channels from whatever value `self.in_ch` was after the final loop iteration, down to a value of `1`, representing a real/fake prediction on the given patch. 
7. Then we take our list of layers, unpack them into a `torch.nn.Sequential`, and store the result in `self.model`.

**The best way to illustrate this is by just looking at what exactly happens to a tensor's shape as it is passed through the network:**
| Layer | Input Shape | Output Shape | Kernel | Stride | Padding | Receptive Field | Notes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
Initial | [1,&nbsp;6,&nbsp;256,&nbsp;256] | [1,&nbsp;64,&nbsp;128,&nbsp;128] | 4x4 | 2 | 1 | 4x4 | Takes channels from `in_channels*2` -> `base_channels` and performs a `4x4` convolution with `stride=2`, halving the tensor's spatial resolution.
`n_layer`&nbsp;#1 | [1,&nbsp;64,&nbsp;128,&nbsp;128] | [1,&nbsp;128,&nbsp;64,&nbsp;64] | 4x4 | 2 | 1 | 10x10 | Doubles channel count, halves spatial resolution.
`n_layer`&nbsp;#2 | [1,&nbsp;128,&nbsp;64,&nbsp;64] | [1,&nbsp;256,&nbsp;64,&nbsp;64] | 4x4 | 1 | 1 | 22x22 | Doubles channel count, but since `stride=1` on this final `n_layer`, spatial resolution remains at `64x64`.
Final | [1,&nbsp;256,&nbsp;64,&nbsp;64] | [1,&nbsp;1,&nbsp;64,&nbsp;64] | 4x4 | 1 | 1 | ***70x70*** | Reduces channel count from `256` (final value from `n_layers` loop) to `1`. And again, since `stride=1`, spatial resolution remains unchanged.
> [!NOTE]
> This example assumes RGB input images (`in_channels=3`) with a resolution of `256x256`, `base_channels=64`, and `n_layers=3`.

**And that's really all there is to the PatchGAN `Discriminator` model.** When fed a set of input tensors, the class's forward function simply concatenates them together along the channel dimension and runs the resulting tensor through the `Sequential` outlined above, then it returns the final prediction map tensor.
## CNNBlock
> **Reference:** [`nectargan.models.patchgan.blocks`](/nectargan/models/patchgan/blocks.py)

Let's have a really quick look at the `CNNBlock` class. This class is super simple, so we won't spend much time on it, but we will briefly touch on what exactly it does. It's basically just a hard coded wrapper which assembles a simple `torch.nn.Sequential` defining a single downsampling layer. The modules in this `Sequential` are:
1. `torch.nn.Conv2d` : A `Conv2d` with `kernel_size=4`, reflection padding of `1`, `bias=False`, and a `stride` equal to the value of the input `stride` argument (decided by the `stride` logic in the `n_layers` loop of the `Discriminator`).
2. `torch.nn.InstanceNorm2d` : Instance normalization.
3. `torch.nn.LeakyReLU` : `LeakyReLU` for activation with a negative slope of `0.2` and `inplace=True`.

The [`forward`](/nectargan/models/patchgan/blocks.py#L24) function is equally simple. All it does is take an input tensor `x`, runs it through this `Sequential`, and returns the result as a tensor.

---