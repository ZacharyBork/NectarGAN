# NectarGAN API - Trainers
> [*`NectarGAN API - Home`*](/docs/api.md)
#### The [`Trainer`](/nectargan/trainers/trainer.py) class is one of the core components of the NectarGAN API.
## Intro
The NectarGAN API provides a base [`Trainer`](/nectargan/trainers/trainer.py) class which you can inherit from to create your own model-specific trainers. The base class provides a number of convenience functions to speed up the process of building and deploying new trainers. 

### Initializing a `Trainer` (subclass)
First, let's have a quick look at the base `Trainer` [`__init__()` function](/nectargan/trainers/trainer.py#L19).

**We can see that, to initialize a `Trainer`, all that is required is a config.** This will be used to initialize the `Trainer`'s internal [`ConfigManager`](/nectargan/config/config_manager.py), which is where all config information within the `Trainer` is pulled from. But what form that init config takes is up to you. Following is an explanation of what each input type is expecting:
> ###### From [`/docs/api/config.md`](/docs/api/config.md)
> | Type | Behaviour |
> | :---: | --- |
> `str` | This is expecting a string with the system path to a `.json` config file. It will attempt to parse the config file and assign the values, or return various exception types if it fails depending on the reason for the failure.
> `PathLike` | The is expecting an `os.PathLike` object containing the system path to a `.json` config file. It behaves exactly as the same as `str` does.
> `dict[str,Any]` | This is expecting a pre-parsed config file (i.e. `with open('config.json', 'r') as f: data = json.load(f)`).<br><br>It will first run a key check on the input vs. the default config, and if the input passes, it will simply assign the values directly to the `ConfigManager`'s `raw` config data, then parse them into the dataclasses. This one is a little strange because it technically bypasses the JSON stuff altogether, which is actually why it exists, to allow the Toolbox to just dump [`Config`](/nectargan/config/config_data.py#L148)-like data directly into a `ConfigManager`, rather than needing an intermediate > `.json` export step.
> `None` | If `input_config` is `None` (default), the `ConfigManager` will attempt to load the [default config file](/nectargan/config/default.json) located at: `/nectargan/config/default.json`. If it is unable to do so, it will raise an exception. 

**However, one thing to note is that, when initializing a `Trainer`, you may also pass it a pre-constructed `ConfigManager`. In this case, the `Trainer` will just replace its internal `ConfigManager` with the one it's passed.**

When a `Trainer` subclass is initialized, the base `Trainer` class will perform a series of setup functions:
1. It will initialize a number of core member variable, all of which will be discussed in more detail below.
2. Take the input config, whatever form that might take, and use it to initialize its internal ConfigManager.
3. Extract the selected device from the config's `config.common.device` and assign it to `self.device` since it's needed so frequently.

**If the trainer was initialized with `quicksetup=False`, then the `Trainer` init function ends here.** However, if `quicksetup=True` (default), the `Trainer` init performs a few extra convenience steps. This will be expanded upon in a later section but, generally speaking, a `Trainer` will almost always be initialized with `quicksetup=True`. For now, though, these are the additional steps taken if `quicksetup=True`:

4. The `Trainer` will build an output directory for the current experiment based on the output and experiment settings in the config. Or if `continue_train` is `True` in the `config` it was passed, it will instead just overwrite its `self.experiment_dir` with the path to the selected experiment to continue.
5. It will then initialize a [`LossManager`](/docs/api/losses/lossmanager.md) to manage all of the losses during training.
6. After that, it will export a copy of its current config to the experiment directory.
7. Then it will check the input `config` to see if `enable_visdom` is `True`. If it is, it will run [`init_visdom()`](/nectargan/trainers/trainer.py#L149) to build and start a Visdom client. See [here](/docs/api/visdom.md) for more information.
## Member Variables
**The base `Trainer` class has a few important member variables to be aware of, some of which are expected to be set by the class's which inherit from it.** Here is each, with a short description:
| Variable | Description |
| :---: | --- |
`self.log_losses` | This is a boolean variable which is set based on the value of the `log_losses` input argument in `Trainer.__init__()`. It defines whether losses run in the context of the given `Trainer` instance should be cached and occasionally dumped to the loss log (see [here](/docs/api/losses/lossmanager.md) for more info).
`self.current_epoch` | An integer value which keeps track of the current epoch. At init, this is `None`. Then, each time the `Trainer`'s [`train_paired()`](/nectargan/trainers/trainer.py#L353) function is called, the value which is passed for the `epoch` argument first has `1` added to it, then it is assigned to `self.current_epoch`. This `+1` operation is just so that you can pass the raw iter variable from the training loop, and still have the first epoch be called `Epoch 1` everywhere where that would be relevant, like strings printed to the console for example.
`self.last_epoch_time` | A floating point variable used to keep track of the time each epoch takes. The time is checked at the beginning and end of each epoch (in [`Trainer.train_paired()`](/nectargan/trainers/trainer.py#L353)), then the start is subtracted from the end and the results is assigned to this variable.
`self.train_loader` | A `torch.utils.data.DataLoader` representing the training dataset. This is expected to be set by the `Trainer` subclass, although the base `Trainer` class provides a utility function to perform the heavy lifting. See [here](/nectargan/trainers/pix2pix_trainer.py#L109) and [here](/nectargan/trainers/trainer.py#L217) for more info.
`self.val_loader` | Exactly the same as `self.train_loader`, but for the validation dataset.
## Member Functions
What follows is an exhaustive list of all of the member functions of the `Trainer` class, broken down by category. A brief description of each will be provided. For a more complete decription, please see the docstring for the given function, which can be accessed here by clicking on the function name.

### Initialization Functions
| Function | Description |
| :---: | --- |
[`init_config`](/nectargan/trainers/trainer.py#L68) | Initializes the `Trainer`'s internal `ConfigManager`.
[`build_output_directory`](/nectargan/trainers/trainer.py#L91) | Builds and experiment output directory based on the current config settings (or assigns the current experiment directory to `self.experiment_dir` if continuing a train on an existing model).
[`export_config`](/nectargan/trainers/trainer.py#L138) | Exports a copy of the `Trainer`'s current config to a `.json` file in the experiment directory.
[`init_visdom`](/nectargan/trainers/trainer.py#L149) | Initializes a Visdom client for the training session.
[`load_checkpoint`](/nectargan/trainers/trainer.py#L159) | Initializes a given network and optimizer with pre-trained weights from a checkpoint file contained within the current experiment directory.
[`quicksetup`](/nectargan/trainers/trainer.py#L57) | Performs a sequence of other init functions to quickly set up all infrastructure needed for training.

### Component Builders
| Function | Description |
| :---: | --- |
[`init_loss_manager`](/nectargan/trainers/trainer.py#L196) | Initializes a `LossManager` to manage losses for the `Trainer` during training.
[`build_dataloader`](/nectargan/trainers/trainer.py#L217) | Initializes a `torch.utils.data.DataLoader` from a [`PairedDataset`](/docs/api/dataset.md) based on settings in the `Trainer`'s current config.
[`build_optimizer`](/nectargan/trainers/trainer.py#L243) | Simple helper function initialize an optimizer for a given network.

### Training Callbacks
***Training callbacks are a core feature of training with the NectarGAN API. They are exremely flexible override methods that allow you to define complex training loops for your custom `Trainer` class with just a few simple function. See [here](/docs/api/trainers/building_a_trainer.md) for more information.***
| Function | Description |
| :---: | --- |
[`on_epoch_start`](/nectargan/trainers/trainer.py#L262) | Run at the very beginning of an epoch in [`Trainer.train_paired()`](/nectargan/trainers/trainer.py#L353), before the training loop begins.
[`train_step`](/nectargan/trainers/trainer.py#L282) | Run once per batch in [`Trainer.train_paired()`](/nectargan/trainers/trainer.py#L353) -> [`Trainer._train_paired_core()`](/nectargan/trainers/trainer.py#L337).
[`on_epoch_end`](/nectargan/trainers/trainer.py#L313) | Run at the very end of an epoch in [`Trainer.train_paired()`](/nectargan/trainers/trainer.py#L353), after all the batches for the epoch have been completed.

### Training Loop
| Function | Description |
| :---: | --- |
[`_train_paired_core`](/nectargan/trainers/trainer.py#L337) | The core `Trainer` paired training loop, called by [`Trainer.train_paired()`](/nectargan/trainers/trainer.py#L353) after it has completed it's epoch init steps and run the `on_epoch_start` callback.
[`train_paired`](/nectargan/trainers/trainer.py#L353) | **This is the core paired adversarial training function.** It largely serves as a public wrapper around `_train_paired_core`, however, it also performs some additional epoch init steps and runs the `on_train_start` callback before the main loop, and the `on_train_end` callback after it is complete.

### Console Logging
| Function | Description |
| :---: | --- |
[`print_end_of_epoch`](/nectargan/trainers/trainer.py#L428) | A function, meant to be run at the end of each epoch, which will print the time that the epoch took to complete to the console. The `Pix2pixTrainer` also [extends this function](/nectargan/trainers/pix2pix_trainer.py#L197) to print a couple extra lines related to learning rate changes for G and D.

### Model/Example Image Saving
| Function | Description |
| :---: | --- |
[`export_model_weights`](/nectargan/trainers/trainer.py#L443) | A utility function which will take a network and its associated optimizer, and save a checkpoint for the network as a `.pth.tar` to the current experiment directory, tagged with the current epoch value at the time the function was called.
[`save_xyz_examples`](/nectargan/trainers/trainer.py#L476) | This function will swap a given network over into eval mode, select a random set of images from the `self.val_loader` dataset (the number of images is defined by the value of ['config']['save']['num_examples'] in the config file), run the network's inference on each image, and for each image, export 3 `.png` files to the current experient directory's example subdirectory. Then it switches the network back in to train mode.
## Building a New Trainer
**The base `Trainer` class can be easily inherited from to create new `Trainer` subclasses for additional models.**

*See [here](/docs/api/trainers/building_a_trainer.md) for more information.*

---