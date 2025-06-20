# NectarGAN API - Building a Trainer
> [*`NectarGAN API - Home`*](/docs/api.md)
#### The base `Trainer` class in the NectarGAN API is intended to be inherited from to allow for the easy creation of new `Trainer` subclasses for models with different requirements. 
###### Strap in, this one is dense :)

> [!NOTE]
> Currently, the `Trainer` class, and the API as a whole, are intended specifically for paired adversarial training (Pix2pix-style[^1]). This is planned to be expanded to unpaired training in the future but it is worth noting that in it's current state, the framework is mostly suited for paired image translation models.

> [!IMPORTANT]
> This section will largely build upon topics addressed in the [documentation](/docs/api/trainers/trainer.md) for the base `Trainer` class. It is advisable to read that before document before beginning this one.

## The `Pix2pixTrainer` class
The [`Pix2pixTrainer`](/nectargan/trainers/pix2pix_trainer.py) serves as an example implementation of a `Trainer` subclass, a Pix2pix-style[^1] GAN model in this case.

Here, we will walk step-by-step though the implementation, to better understand how it works and how we can implement trainers for other models in the same style.

### The Pix2pixTrainer's `__init__()` Function
Starting with our input arguments for  `Pix2pixTrainer.__init__()`, we can see that, like the base `Trainer` class, it requires some sort of config object. Makes sense, we need to pass it to the parent `Trainer` for its init function. We can also see that it takes a `log_losses` argument for the same reason. It also takes a `Literal` string called `loss_subspec`. This is expanded upon in much greater detail [here](/docs/api/losses/loss_spec.md) but in short, this tells the `Pix2pixTrainer`'s internal `LossManager` what loss functions to register when it is initialized.

Then, the first thing we do is initialize the parent `Trainer` class, passing it the input `config` and the `log_losses` boolean. We also pass it a value of `True` for `quicksetup`, telling the `Trainer` that we would like it to automatically build an experiment output directory and export a copy of the config as a `.json` to it, a `LossManager`, and a Visdom client, if applicable based on the current config.

After that, we run a sequence of init functions. In order, these are:
| Function | Description |
| :---: | --- |
[`_init_lr_scheduling`](/nectargan/trainers/pix2pix_trainer.py#L51) | Checks the value of `separate_lr_schedules` in the current config. If it is `True`, it does nothing. If it is `False`, however, it will override the discriminator's learning rate with the value of the generator's learning rate, allowing learning rates for the discriminator to be accessed the same way, regardless of whether it is using a separate schedule.
[`_init_generator`](/nectargan/trainers/pix2pix_trainer.py#L57) | Initializes a [`UnetGenerator`](/docs/api/models/unet.md) network, and a `torch.nn.optim.Adam` optimizer for the network.
[`_init_disriminator`](/nectargan/trainers/pix2pix_trainer.py#L84) | Initializes a PatchGAN-style [`Discriminator`](/docs/api/models/patchgan.md) network, and a `torch.nn.optim.Adam` optimizer for the network.
[`_init_dataloaders`](/nectargan/trainers/pix2pix_trainer.py#L109) | Initializes one `torch.nn.data.DataLoader` each for the `train` and `val` dataset.
[`_init_gradscalers`](/nectargan/trainers/pix2pix_trainer.py#L114) | Initializes a gradient scaler for both G and D based on the device in the current config.

**Once those are completed, we initialize our losses with the `Trainer`'s `LossManager`.** This is done here by running the function [`Pix2pixTrainer._init_losses()`](/nectargan/trainers/pix2pix_trainer.py#L122). You are encouraged to read the docstring for this function as it is extremely thorough, but in short, it takes the `loss_subspec` argument which was passed to the `Pix2pixTrainer` at init, validates it against a list of supported subspecs, and then passes that along with the input `config` and the a reference to the NectarGAN (see [here](/docs/api/losses/loss_spec.md) and [here](/nectargan/losses/pix2pix_objective.py)) to the `init_from_spec()` function of the `Trainer`'s internal [`LossManager`](/docs/api/losses/lossmanager.md). This registers all the losses required loss functions for training.

**Finally, we do a quick check of the current config to see if we are continuing a train on a pre-existing model.** If we are, we run the [`load_checkpoint()`](/nectargan/trainers/trainer.py#L159) function from the parent `Trainer` class to load the weights for both the generator and the disriminator.

**And that's it really, our `Trainer` subclass is basically ready for training, save for one small thing...**
## Training Callbacks
#### At the heart of how the `Trainer` class and its subclasses function are the three main training loop override methods.
---
Here are the three methods with a short description:
> ###### From [`/docs/api/trainers/trainer.md`](/docs/api/trainers/trainer.md)
> | Function | Description |
> | :---: | --- |
> [`on_epoch_start`](/nectargan/trainers/trainer.py#L262) | Run at the very beginning of an epoch in [`Trainer.train_paired()`](/nectargan/trainers/trainer.py#L353), before the training loop begins.
> [`train_step`](/nectargan/trainers/trainer.py#L282) | Run once per batch in [`Trainer.train_paired()`](/nectargan/trainers/trainer.py#L353) -> [`Trainer._train_paired_core()`](/nectargan/trainers/trainer.py#L337).
> [`on_epoch_end`](/nectargan/trainers/trainer.py#L313) | Run at the very end of an epoch in [`Trainer.train_paired()`](/nectargan/trainers/trainer.py#L353), after all the batches for the epoch have been completed.

Looking at each of the three functions, a few things stand out:

1. In their implementation in the base `Trainer` class, all they do is raise an exception. This means that any `Trainer` subclass is expected to override all three methods, otherwise they will raise an exception during the training loop.
2. `on_epoch_start` and `on_epoch_end` have no required arguments apart from a non-descript dict of `kwargs`. This indicates here that they don't do anything on their own, and that they don't have any ***required*** functionality in a `Trainer` subclass. We will touch on this more in a second.
3. `train_step` is different. It does still accept the `kwargs` dict like the other two, but it also has 3 required arguments, all of which are passed to it during the training loop in [`Trainer.train_paired()`](/nectargan/trainers/trainer.py#L353) -> [`Trainer._train_paired_core()`](/nectargan/trainers/trainer.py#L337). These are:
    - `x`: A `torch.Tensor` representing the input(s) for the current batch.
    - `y`: A `torch.Tensor` representing the ground truth(s) for the current batch.
    - `idx` : An integer representing the current train loop interation at the time the function was called, indexed from zero.
---
**Now let's have a look at the function that executes all three of the callbacks, [`Trainer.train_paired()`](/nectargan/trainers/trainer.py#L353), to better contextualize how they function.** 

Looking over the function, we first notice that it takes a slightly strange set of input arguments. Let's start by going through each one to see what it does:
| Argument | Description |
| :---: | --- |
`epoch` | The iteration value of the training loop at the time this function is called. Used to set the `Trainer`'s `current_epoch` value (i.e. `Trainer.current_epoch == 1+epoch`)
`on_epoch_start` | A `Callable`. Run once, right before the training loop begins.
`train_step` | A `Callable`. Run once per batch in the dataset.
`on_epoch_end` | A `Callable`. Run once, after all batches have been completed.
`multithreaded` | If True (default), this function will start a new thread to update the Visdom visualizers. If False, it will update them in the same thread used for training.
`callback_kwargs` | A dict[str, dict[str, Any]] with any keyword arguments you would like to pass to the callback functions during training. kwargs are parsed internally and passed to their repective callbacks. The dict should be formatted as follows:<br><br>callback_kwargs = {<br>&nbsp;&nbsp;&nbsp;&nbsp;'on_epoch_start': { 'var1': 1.0 },<br>&nbsp;&nbsp;&nbsp;&nbsp;'train_step': { 'var2': True, 'var3': [1.0, 2.0] },<br>&nbsp;&nbsp;&nbsp;&nbsp;'on_epoch_end': {'var4': { 'x': 1.0, 'y': 2.0 } }<br>}<br><br>See `Pix2pixTrainer.on_epoch_start()` for example implementation.

**Looking at these arguments, we can see that three of them have the same names as our training callbacks.** If we quickly jump down to where we [initialize our training functions](/nectargan/trainers/trainer.py#L401), we can see that for each of the training loop functions (`start_fn`, `train_fn`, `end_fn`), we first check to see if the input argument for that value is `None`. If it's not, we use the `Callable` that was passed as an input to `Trainer.train_paired()`, but if it is (default), we instead use the corresponding callback function. This allows you to technically bypass the callback override methods altogether. More realistically, though, this can be used to quickly drop in a function for one of the steps to test a change non-destructively.

**Next, let's have a quick look at the `callback_kwargs`.** As we touched on previously, all three callback methods take a dict of kwargs. These kwargs are first passed to `Trainer.train_paired()` as one large dict, formatted as shown in the table above, where each callback is a subdict in the main input `callback_kwargs` dict, whos key is the name of the function, and who's values can be... well, whatever you want really. Let's have a quick look at how this is used. We will use the core paired training script and the `on_epoch_start` method from the `Pix2pixTrainer` as an example:

> ##### From [`/scripts/paired/train.py`](/scripts/paired/train.py)
> ```python
> trainer = Pix2pixTrainer(
>     config=args.config_file, 
>     loss_subspec=args.loss_subspec,
>     log_losses=args.log_losses)
> 
> for epoch in range(epoch_count):
>     trainer.train_paired( # Train generator and discriminator
>         epoch, 
>         callback_kwargs={ 'on_epoch_start': {'print_train_start': True} }) 
> ```

Looking at the training script, we can see that, when we called the `train_paired` function on our `Pix2pixTrainer`, we pass it a dictionary, structured as follows:
```python
{ 
    'on_epoch_start': {
        'print_train_start': True
    } 
}
```

Now, let's have a look at how that dict is handled by the `Trainer` inside of the `train_paired` function when we pass it to the `callback_kwargs` argument:
> ##### From [`nectargan.trainers.trainer.Trainer.train_paired`](/nectargan/trainers/trainer.py#L)
> ```python
> [...]
>
> start_fn = on_epoch_start or self.on_epoch_start # Init pre-train fn
> 
> [...]
>
> start_fn(**callback_kwargs.get('on_epoch_start', {})) 
>
> [...]
> ```

Stripping it down to just the relevant couple lines here, we can see that the `train_paired` first assigns the current `start_fn`, as described above. Then later on, it calls that function and, when doing so, gets the `on_epoch_start` dict from the input `callback_kwargs` dict (or an empty dict if there were no `on_epoch_start` dict), unpacks it into keyword arguments, and passes them as the `kwargs` argument. This same process is also done for `train_step` and `on_epoch_end`.

Finally, let's have a brief look at how to use our `print_train_start` kwarg inside of our `on_epoch_start` callback. There are a couple ways to do this, so first, we'll see how it is done in the `Pix2pixTrainer`, and then we will see an alternative method, then we will discuss why the `Pix2pixTrainer` does not use this alternative method, despite them both being equally valid.

First, the `Pix2pixTrainer`'s `on_epoch_start` callback:
> ##### From [`nectargan.trainers.pix2pix_trainer.Pix2pixTrainer`](/nectargan/trainers/pix2pix_trainer.py#L428)
> ```python
> def on_epoch_start(self, **kwargs: Any) -> None:
>     if 'print_train_start' in kwargs.items():
>         if kwargs['print_train_start']:
>             print(f'Beginning epoch: {self.current_epoch}')
>             self.loss_manager.print_weights()
> ```
Looking at our callback method, we can see that it treats the incoming `kwargs` argument as a dict. It first checks the items of the dict to see if `print_train_start` is present and, if it is, it then checks to see if the value is `True` or `False`. It it is `False`, nothing happens. If it is `True`, though, it will print a short string showing the index of the epoch which is about to begin, and then calls its `LossManager`'s [`print_weights`](/nectargan/losses/loss_manager.py#L342) to print all of the loss weights which will be used in the upcoming epoch.

Now, an alternative method:
```python
def on_epoch_start(self, print_train_start: bool) -> None:
    if print_train_start:
        print(f'Beginning epoch: {self.current_epoch}')
        self.loss_manager.print_weights()
```
We can see that the core functionality is exactly the same, they both print the same values based on the same input `print_train_start` conditional, but where the method used by the `Pix2pixTrainer` treats the incoming `kwargs` as a dict, this one takes advantage of the fact that the `Trainer`'s `train_paired` function unpacks the relevant subdict as it passes it to the callback to instead explicitly define the callback method's `print_train_start` argument.

The `Pix2pixTrainer` chooses the first method in an effort to be more open-ended. That method makes the `print_train_start` argument effectively optional, whereas the second method would raise an exception if the kwarg was not passed when calling `train_paired`.

**Both approaches are totally valid though. Ultimately, they both acchieve the same goal.**

---
**Lastly, let's have a look at arguably the most important of the three training callbacks, `train_step`.**

This method houses your core training function (i.e. forward and backward steps for G and D). It takes as input two `torch.Tensors`, `x`, the real input image(s) from the current batch of the dataset when the function is called, and `y`, the real ground truth image(s) from the current batch. It also takes an integer, representing the current training loop iteration at the time the function is called.

Let's take a quick look at the `train_step` implementation in the `Pix2pixTrainer` to better understand exactly how it works.
> ##### From [`nectargan.trainers.pix2pix_trainer.Pix2pixTrainer`](/nectargan/trainers/pix2pix_trainer.py#L472)
> ```python
> def train_step(
>         self, 
>         x: torch.Tensor, 
>         y: torch.Tensor, 
>         idx: int,
>         **kwargs: Any
>     ) -> None:
>     with torch.amp.autocast('cuda'): 
>         y_fake = self.gen(x)
> 
>     # Get discriminator losses, apply gradients
>     losses_D = self.forward_D(x, y, y_fake)
>     self.backward_D(losses_D['loss_D'])
>     
>     # Get generator losses, apply gradients
>     loss_G = self.forward_G(x, y, y_fake)
>     self.backward_G(loss_G)
>     
>     if idx % self.config.visualizer.visdom.update_frequency == 0:
>         self.update_display(x, y, y_fake, idx)
> ```
Looking at our `train_step` method, we can see what looks like a single step of a very standard Pix2pix training loop. We run the generator's inference to create a y_fake, calculate and apply discriminator losses, then do the same for the generator. Then we have a small function to update the display (Visdom and console), if it is applicable this batch. 

**That is all the `train_step` callback is. Just a single step of the model's training loop.**

---
### Recap
**Alright, so we've seen our three different callback functions, and we've had a quick look at the [function that invokes them](/nectargan/trainers/trainer.py#L353). Now let's briefly recap each, and have a quick look at what is does, how and where exactly each is used in the base `Trainer`, and how the `Pix2pixTrainer` handles overriding each method, so that we can better understand how they can be used to quickly prototype training loops for our other models.**
### `on_epoch_start`
**Description:** This callback is run once at the [very beginning of the epoch](/nectargan/trainers/trainer.py#L408), before the actual training loop is started.

**Use:** The `Pix2pixTrainer` uses this method to print some useful data to the console (i.e. loss weights and current epoch value). 
### `train_step`
**Description:** Run once per batch during each epoch, not directly by the `train_paired` function, but instead via the private [`_train_paired_core`](/nectargan/trainers/trainer.py#L337)

**Use:** Forward and backward train step.
### `on_epoch_end`
**Description:** This callback is run once at the [very end of the epoch](/nectargan/trainers/trainer.py#L408), after all of the batches for that epoch have been completed.

**Use:** The `Pix2pixTrainer` uses this method to dump cached loss values to the `.json` log and to update the models schedulers.

[^1]: [Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks (Isola et al., 2017)](https://arxiv.org/abs/1611.07004)