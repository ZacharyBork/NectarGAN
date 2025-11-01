# NectarGAN API - Loss Specs
> [*`NectarGAN API - Home`*](../../api.md)
#### Loss specifications (specs) are a novel way to define reusable objective functions for your models. At their core, loss specs are just standard Python functions but with one specific requirement, the must return a dictionary of string-mapped `LMLoss` objects.
## What is a loss spec?
To better explain the concept, let's first have a quick look at an example loss spec function: [`nectargan.losses.pix2pix_objective`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/losses/pix2pix_objective.py)

Let's quickly break down exactly what this function is doing:

1. We grab the current device from the config so that we can cast our loss functions to it.
2. We instantiate the core losses from the original Pix2pix paper[^1], BCEWithLogits for cGAN, L1 for G pixel-wise loss (casting each to our current device).
3. We create a dictionary, and add our base loss functions.
    - We set the dict key for each to the name we want to use for lookup (equivalent to `LossManager.register_loss_fn(loss_name='loss_fns_dict_key')`).
    - We instantiate an `LMLoss` object, setting the name to the same value as the dict key, the function to the `nn.Module` of the associated loss function (i.e. 'G_GAN'=BCE, 'G_L1'=L1, etc.).
    - We assign loss weights, if applicable. `D_real` and `D_fake` do not have lambda functions so we do not set the value of the `loss_weight` argument for either, giving them the default weight of `1.0`.
    - We assign tags for each. In this spec, the tags are utilized to differentiate which network the given loss belongs to, so that they can be looked up by group for Visdom/Toolbox visualization. You can have as many tags as you want and use them for whatever you want though.
4. Then, we do a couple checks on the value of the `subspec` argument. If the requested subspec is `extended`, we also add `MSE`, `Sobel`, and `Laplacian` losses to the dictionary, and if `+vgg` was in the `subspec` value, we also add `VGGPerceptual`. 
> [!NOTE]
>`VGGPerceptual` is in a separate category here because, the first time a user runs it, it will automatically download the VGG19-basic weights from Torch.
5. We return the dictionary of LMLoss'.

Pretty simple, right? Just a function which defines a dict containing all the losses you want to use for your model. Now let's have a look at how they're used. First, we will create a `LossManager` instance: 

```python
from nectargan.config.config_manager import ConfigManager
from nectargan.losses.loss_manager import LossManager

config_manager = ConfigManager('path/to/config.json')
loss_manager = LossManager(
    config=config_manager.data,
    experiment_dir='/path/to/experiment/output/directory')
```
Should look pretty familiar from the first section. However, in the first section, we then went on to register a loss function with `loss_manager.register_loss_fn()`. And that is fine for quickly building model objectives or experimenting with different losses, but now we have a known objective that we want to use, and a function that defines it. So, instead of registering all of our losses one by one, now, we can instead just do this:
```python
import nectargan.losses.pix2pix_objective as spec # Import our loss spec function
loss_subspec = 'extended+vgg' # Define a subspec, since our loss spec expects one

loss_manager.init_from_spec(  # Run 'LossManager.init_from_spec()'
    spec.pix2pix,             # Pass it a reference to our loss spec function,
    config_manager.data,      # our config data, since the spec is expecting it,
    loss_subspec)             # and our string defining the subspec
```
And that's it, now the `LossManager` will run that function, parse the returned dict, and register all of the losses contained within it, then you are free to use them how you would with any other loss registered with a `LossManager` instance. And you also now have a loss spec which can be reused whenever you'd like to perfectly replicate that objective function.

Now let's have a quick look at [`LossManager.init_from_spec()`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/losses/loss_manager.py#L116) to see what it's doing. It serves one main purpose, setting the value of the `LossManager`'s `self.loss_fns`. Let break down exactly how it does that, though:

1. We have a few input arguments, one required argument, `spec`, which takes a `Callable` (our loss spec function), and also optional `*args` and `**kwargs`. **This allows you to pass whatever extra arguments you want to your loss spec function. This is how the `pix2pix_objective` loss spec is passed the `config` and `subspec`.** ***These are not required arguments. The only requirement for a loss spec is that is be a `Callable` that returns a `dict[str, LMLoss]`.***
2. We first do a quick check to see if the value of the `spec` argument is `None`. If it is, we just set `self.loss_fns` to an empty dict. This is done when the `LossManager` if first instatiated to initialize it's base loss dict, which is what losses are added to when you call `LossManager.register_loss_fn()`.
3. If `spec` is not `None`, however, we then try to run the loss spec function, then we loop through all the values in the spec function's return dict and assert that they are `LMLoss` instances.
4. If the return dict values check out, we then assign the return values to `self.loss_fns` and run a helper function which creates dummy tensors for each `LMLoss`'s `last_loss_map`.

## Summary
#### **So, in short, loss specs let you pre-define model objectives quickly, and in a format that encourages reproducability. From a technical standpoint, they are just Python functions that return a `dict[str, LMLoss]`. What arguments those functions take, though, and how they construct that return value is entirely up to you. They are extremely open-ended.**
[^1]: [Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks (Isola et al., 2017)](https://arxiv.org/abs/1611.07004)