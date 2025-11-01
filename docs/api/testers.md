# NectarGAN API - Testers
> [*`NectarGAN API - Home`*](../api.md) [*`Getting Started (Toolbox | Testing)`*](../getting_started/toolbox_testing.md)

#### Nectargan provides a `Tester` class and a paired testing helper script to allow you to test your trained models on real input data.
## The `Tester` class
> **Reference:** [`nectargan.testers.tester`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/testers/tester.py)

> [!NOTE]
> Right now, as there is only one type of model trainer (for Pix2pix-style paired image translation), the `Tester` class is only suitable for paired image translation model testing.
>
> Support for unpaired models is planned for the future, and when it is added, this `Tester` will be refactored into a base `Tester` class, and child classes for paired and unpaired testing, as it is with the `Trainer`/`Pix2pixTrainer`. 

Let's start by having a look at the `Tester` class.

The first thing we notice is that it inherits from the base [`Trainer`](../api/trainers/trainer.md) class. The `Tester` uses its parent class to:
- Manage configuration and output settings
- Handle model checkpoint loading.
- Build a dataloader for testing.

### Initializing a `Tester`
Looking at the `__init__` method, we can see that a `Tester` can take up to four possible input arguments:
| Argument | Description |
| :---: | --- |
`config` | This can be any number of things representing a config (see [here](../api/config.md)), or `None` (default). This config is primarily used to define generator architecture settings, although it may also be used for a few other things depending on the other arguments which are passed to it.<br><br> **If the generator you are loading for testing uses different architecture settings than the default config, you will need to pass a config (like the default train config, exported automatically for each train). Otherwise you can use the default `None`**
`experiment_dir` | The experiment directory containing the checkpoints of the model you would like to test.<br><br> **_This should almost always be set._ It defaults to `None` for Toolbox compatibility reasons, but the process of actually using that `None` value is complicated. You will likely get exceptions if you try to use the default.** 
`dataroot` | An `os.PathLike` object pointing to the root directory of a dataset, or `None` (default). The `dataroot` directory must contain a subdirectory called `test` containing the dataset images to use for testing. If this value is `None`, the dataroot from the config will be used instead.
`load_epoch` | The epoch checkpoint to load for testing (i.e. `epoch{load_epoch}_netG.pth.tar`), or `None` (default). If this value is `None`, the value of `["config"]["train"]["load"]["load_epoch"]` from the provided config will be used instead.

**Now let's walk step by step through the function to see exactly what happens when we initialize a `Tester`:**
1. We initialize the parent `Trainer` class, passing it the input config, and disabling the quicksetup since we don't need many of the components which are required for training.
2. Force the `continue_train` boolean value in the config to `True`. We do this so that we can call the parent class' [`build_output_directory`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/trainers/trainer.py) method to initialize the current `self.experiment_dir` (*a member variable from the parent `Trainer` class which defines the current experiement directory*) path without it raising an exception. See [here](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/testers/tester.py#L79) for more info. 
3. Override the config load epoch if one was passed as an input argument.
4. Then check if the input `experiment_directory` is `None`:
    - **If it is:** We run [`Tester._build_output_directory()`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/testers/tester.py#L79). This calls `Trainer.build_output_directory()`, as discussed above, then calls [`Tester._init_test_output_root()`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/testers/tester.py#L65). All this function does is perform a quick check of the current `experiment_directory` to see if it contains a subdirectory called `test`. If not, it tries to make one.
    - **If it isn't:** We just set `self.experiment_dir` to the input `experiement_directory`, then call [`Tester._init_test_output_root()`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/testers/tester.py#L65).
5. Override the config `dataroot` if one was passed.
6. Initialize the `test` dataloader and the generator. 

**So then, with these input arguments in mind, let's see how we can initialize a `Tester` instance:**
```python
from nectargan.testers.tester import Tester

tester = Tester(
    config='/path/to/config_file.json',
    experiment_dir='/path/to/experiment/directory',
    dataroot='/path/to/dataset/root',
    load_epoch=200
)
```
**And that's all there is to it. After that, we're ready for...**
### Testing
> **Reference:** [`Tester.run_test`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/testers/tester.py#L258)

With our `Tester` instance created, we can now perform a test like so:
```python
tester.run_test(image_count=10)
```
This will tell the tester to select 10 random images (as defined by the `image_count` argument) from the `test` dataset. It will then create a new output directory for the test inside of `experiment_directory/test` and, inside of that output directory, create a base `.json` log to dump test data to. 

Then, for each image, it will perform a number of steps:
1. Run the generator's inference on the input image.
2. Evalute the generator's output using a few loss function (currently [`L1`](https://docs.pytorch.org/docs/stable/generated/torch.nn.L1Loss.html), [`Sobel`](../api/losses/loss_functions.md), [`Laplacian`](../api/losses/loss_functions.md))
3. Save the [`x`, `y_fake`, `y`] image set to the test output directory.
4. Write the data from that test iteration (i.e. input image path, output image paths, loss values, etc.) to the test log.

> [!NOTE] 
> Sampling from the `test` dataset is random, so if you have more `test` images than you initially set `image_count` to, running it again will pick a new random set of images to test on.


## Paired Testing Script
> **Reference:** [`nectargan.start.testing.paired`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/start/testing/paired.py)

NectarGAN provides a prebuild script for interacting with the `Tester`, allowing you to run model tests from the command line. For more information, please see [here](../getting_started/cli.md).

## Toolbox Testing
Model testing in the Toolbox is currently much more "built-out", so to speak. Naturally, it is deeply integrated in to the interface, but it may serve as a good example of a more complex testing integration, so we will discuss it briefly here.

**Toolbox testing is facilitated by two core components:**
1. [**`TesterWorker`**](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/toolbox/workers/testerworker.py)

    A worker class which performs the actual testing inside of a `QThread`. `TesterWorker` is a child of `Tester`, and uses basically all of its functionality wholesale. The only difference is that it implements its own version of `Tester.run_test()`, [`TesterWorker.run()`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/toolbox/workers/testerworker.py#L29).

    `run` is effectively just a `QThread` compliant override of `Tester.run_test()`. Where `run_test` prints progress updates to the console, `run` instead emits a `progress` signal. And when all the test iterations are complete, it also emits a `finished` signal. These signals are picked up and processed by the...

2. [**`TesterHelper`**](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/toolbox/helpers/tester_helper.py)

    A helper class which handles all things related to model testing in the NectarGAN Toolbox. It sets up `TesterWorkers`, manages their `QThreads`, processes signals during testing, and loads the resulting data when testing is complete.
    
    We won't go too deep in to this class, as most of its functionality is really only pertinent to the UI. There are a couple functions worth noting, though, for more general handling of test results.

    - [`_parse_test_log`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/toolbox/helpers/tester_helper.py#L223)

        This function serves as an example of programmatic loading of the test log generated by the tester, and extraction of just the test results for each iteration (the base test log includes some additional metadata, not strictly related to the test itself).

    - [`_sort_results`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/toolbox/helpers/tester_helper.py#L102)

        Called in `_parse_test_log`, this function is used to sort the results from the test log by a number of metrics from the logs.
        
    - [`load_test_results`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/toolbox/helpers/tester_helper.py#L233)

        This function is very much tied to PySide, but it serves as an abstract example of iterating through the test results from the log and extracting the paths to the example images exported during testing.

---