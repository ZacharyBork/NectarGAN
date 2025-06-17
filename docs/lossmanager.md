# NectarGAN - LossManager
#### The loss manager is one of the core features of the NectarGAN API. It is a drop-in solution for managing, tracking, and logging everything related to loss in your model.

Source: [nectargan.losses.loss_manager.LossManager](/nectargan/losses/loss_manager.py)

## Key Features
- **Builds an easy to use wrapper around around any loss function,** allowing you to evaluate loss functions in your training script in a way which is as easy as calling loss functions traditionally, but which dramatically expands the backend functionality of any loss function registered with the `LossManager`.
- **Caches loss function results in multiple formats** with easy to use mechanisms for recalling the values during training.
- **An intelligent cache management system** allows mean loss values to be cached to memory, and dumped to a JSON log at your discretion, or automatically if a configurable cache limit is reached.
- **Quickly initialize a configurable objective function with a pre-built loss spec,** or register your own loss functions with the `LossManager` to build your own model objective from scratch, while still being able to use all the QOL features that the LossManager offers. You can even define your own reusable loss specs to feed to the `LossManager`, and it will take care of the rest. 
## LossManager Dataclasses
To understand how the loss manager functions and how it manages the data for the losses that are registered with it, we first have to take a quick look at two dataclasses which are at the core of it's functionality. These are:

### 1. [`nectargan.losses.lm_data.LMHistory`](/nectargan/losses/lm_data.py)

**Starting with the simpler of the two dataclasses, `LMHistory` only has one job:** ***store previous loss value history.***

Every loss function registered with a `LossManager` instance has an `LMHistory` instance assigned to it in a way which will be explained momentarily. An `LMHistory` instance contains just two lists, they are dual-purpose, however. If loss logging is enabled, i.e. `LossManager(enable_logging=True)`, these two lists will be used to store the mean value of the loss result tensor and the current weight value of the loss, both as 32bit floating point values, every time the parent loss function is called via `LossManager.compute_loss_xy()`.

If logging is disabled, however, each time `LossManager.compute_loss_xy()` is called for a given loss, both lists in that loss's `LMHistory` are cleared, after which time the new values are appending to each list. In practice, this means that if `enable_logging=False` each list will only store a single value, the most recent loss mean and weight respectively, at any given time.

### 2. [`nectargan.losses.lm_data.LMLoss`](/nectargan/losses/lm_data.py)

***This dataclass is responsible for storing all information about a registered loss.*** For every loss function that is registered with a `LossManager` instance via `LossManager.register_loss_fn()`, an `LMLoss` instance is created which describes the loss function. A full description of the values contained with an `LMLoss` instance can be seen by clicking on the above link, but here is a rough outline:

- `name`: a string, unique to this registered loss, which is used for lookup by various `LossManager` functions.
- `function`: a reference to the `torch.nn.Module` for the loss function. This can be almost any `Module` as long as it has a forward function that returns a tensor. One caveat is that, currently, loss functions registered with the `LossManager` can only accept two input tensors for loss computation (`y`, `y_fake`), although I do plan to expand that at some point in the future.
- `loss_weight`: The weight value (lambda) to apply to the resulting loss tensor when it is called, before the tensor is returned by `LossManager.compute_loss_xy()`.
- `schedule`: A [`Schedule`](/docs/scheduling.md) object defining a weight schedule for the given loss. If no `Schedule` is provided when the LMLoss is initialized, the provided `loss_weight` will be used for the duration of training.
- `last_loss_map`: This is not set when initializing an `LMLoss` object. It is instead initialized as a dummy tensor, and then used by the `LossManager` each time the parent loss function is run to store a detached version of the resulting loss tensor so they can be recalled for visualization.
store history.
- `history`: This is also not set at init-time. A unique `LMHistory` object is automatically created and assigned to every loss registered with the loss manager.
- `tags`: An optional list of strings containing identifier tags which can be used to search for and filter registered losses in various `LossManager` functions.

## Using the LossManager
Initializing a new `LossManager` instance:
```python
from nectargan.config.config_manager import ConfigManager
from nectargan.losses.loss_manager import LossManager

config_manager = ConfigManager('path/to/config.json')
loss_manager = LossManager(
    config=config_manager.data,
    experiment_dir='/path/to/experiment/output/directory')
```
Register a new loss function with a `LossManager` instance:
```python
import torch.nn as nn
L1 = nn.L1Loss().to(config_manager.data.common.device)

loss_manager.register_loss_fn(
    loss_name='mylossfunction',
    loss_fn=L1,
    loss_weight=100.0,
    tags=['descriptive_lookup_tag'])
```
> [!WARNING]
> The `loss_name` you assign to your loss function when you register it must be unique amongst all other loss functions registered with that `LossManager` instance. If you attempt to register a loss function with a name that is already registered, the `LossManager` will raise an exception.

Running your loss function via the loss manager will return the result of the given loss function's `forward()` function as a `torch.Tensor`. The Tensor that is return has had weights pre-applied by `LossManager.compute_loss_xy() -> LossManager._weight_loss()`, based on whatever the current weight value of the registered loss is:
```python
import torch

y = torch.Tensor()      # Ground truth
y_fake = torch.Tensor() # Generator output 

result: torch.Tensor = loss_manager.compute_loss_xy(loss_name='mylossfunction', x=y_fake, y=y)
```
## Querying Registered Loss Data
Data relating to losses registered with a given `LossManager` instance can be retrived in a variety of ways, dependent upon exactly what data you are trying to query, and what format you would like it returned in.

### Querying LMLoss Objects Directly
The most flexible method is to just query the raw `LMLoss` objects directly. This can be done as follows:
```python
losses: dict[str, LMLoss] = loss_manager.get_registered_losses(query=None)
```
This will return all registered loss functions as a dict. The key for each loss will be the name it was registered with. So for our above example, we could then query any info related to our `mylossfunction` function like this:
```python
mylossfn: LMLoss = losses['mylossfunction'] # Query the LMLoss object
lossfn = mylossfn.function                  # Get the loss function module
loss_map = mylossfn.last_lost_map           # Get the most recent loss result as a torch.Tensor
```

> [!NOTE]
> There is one thing to be aware of when querying the `LMLoss` objects directly like this. Were you to do this, expecting to get the loss value and weights history lists:
> ```python
> values : dict[str, float] = mylossfn.history.losses
> weights: dict[str, float] = mylossfn.history.weights
> ```
>You would find that `weights` and `values` are empty lists. **This is intentional**. `LossManager.get_registered_losses()` has an optional flag called `strip` which defaults to `True`. If this flag is not overridden with a value false, the losses `history.values` and `history.weights` lists are cleared. 
>
>The reasoning behind this is that dependent upon what the `history_buffer_size` of the `LossManager` is set at, these lists can get fairly long. And if you have a significant amount of registered losses, passing them around can become a fairly heavy task, so they are stripped by default to reduce the memory overhead. If you need these values for whatever reason, though, just call `LossManager.get_registered_losses()` with `strip=False`. As long as the `LossManager`'s `history_buffer_size` is kept to below a reasonable value (i.e. ~100,000), the cost realistically isn't all that concerning.
### Querying Loss Values (as a dict)
Loss values can be retrieved in dictionary form as follows:
```python
values: dict[str, float] = loss_manager.get_loss_values(precision=2)
```
This will return a dictionary of floating point values with lookup keys matching the corresponding loss's name. The optional `precision` argument tells the function how many digits after the decimal point to round the returned loss values to. The default is `2` to keep things tidy if you want to print or otherwise log the values, but you can increase it if you need more precise return values.

> [!IMPORTANT]
> `LossManager.get_loss_values()` uses the last value stored in `LossManager.history` for each loss. As such, this function should generally be called **AFTER** all of the registered loss functions have been run for the batch. Calling it before running some or all of the loss funtions could lead to unexpected results.
### Querying Loss Values (as a tensor)
The most recent loss tensor, which is detached and stored in the loss function's `LMLoss` every time the loss is called with `LossManager.compute_loss_xy()`, can be queried with:
```python
tensors: torch.Tensor = loss_manager.get_loss_tensors()
```

> [!IMPORTANT]
> `LossManager.get_loss_tensors()` uses the last value stored in the `LMLoss` objects `last_loss_map`. As such, this function should generally be called **AFTER** all of the registered loss functions have been run for the batch. Calling it before running some or all of the loss funtions could lead to unexpected results.

### Querying Loss Weights
The current weight value of all registered losses can be retrieved as follows:
```python
weights: dict[str, float] = loss_manager.get_loss_weights(precision=2)
```
This will return a dictionary of floating point values with lookup keys matching the corresponding loss's name. The values will represent the current weight of the loss at the time that the function was run. The optional `precision` argument tells the function how many digits after the decimal point to round the returned weight values to. The default is `2` to keep things tidy if you want to print or otherwise log the values, but you can increase it if you need more precise return values.

> [!IMPORTANT]
> `LossManager.get_loss_weights()` uses the last weight value stored in `LossManager.schedule.current_value`. Since this value is updated immediately after the associated loss has been run, the weight values that this function returns are the ones which will be applied the next time the loss is run.

### Using Tags to Query Registered Losses
All of the above loss querying functions accept an optional argument called `query`, which is a list of strings. This can be used to query loss values by tag; only loss values which have a tag matching one of the strings in the input `query` argument will be returned. For example, when we registered our loss function above, we assigned it a tag of `descriptive_lookup_tag`. Were we to then want to query losses with that tag, it would look something like this:
```python
lossfns: dict[str, LMLoss] = loss_manager.get_registered_losses(query=['descriptive_lookup_tag'])
values : dict[str, float]  = loss_manager.get_loss_values(query=['descriptive_lookup_tag'])
tensors: dict[str, Tensor] = loss_manager.get_loss_tensors(query=['descriptive_lookup_tag'])
weights: dict[str, float]  = loss_manager.get_loss_weights(query=['descriptive_lookup_tag'])
```
## Dumping Cached Values to Loss Log

> [!NOTE]
> This section is only applicable if the given `LossManager` instance was intialized with `enable_logging=True`.

When a `LossManager` instance is initialized, and optional argument can be passed called `history_buffer_size`. This value defines how many values (stored as 32bit floating point values) can be stored in ***each list*** (i.e. `losses`, `weights`) of ***each LMLoss's*** LMHistory container. By default, this is `50000`. So, with the default value, each registed loss is allowed to store 50,000 unique previous loss values and 50,000 unique previous loss weight values. 100,000 32bit floats per registered loss. This is a totally acceptable fallback value on any modern system, but you are also welcome to set the value higher when you initialize the `LossManager`.

Whenever any registered loss is run from any `LossManager` instance, the `LossManager` first does a quick check to see if the buffer for that loss is full. If it is, it will dump the loss's buffer to the log, clear the buffer, then run the loss function it was originally going to run and appends the result to the now freed up buffer. This is a fine way to handle loss logging with the `LossManager`. It is set and forget, just give it a value for the `history_buffer_size` (or don't even, and just use the default `50000`) and the `LossManager` will take care of the memory management from there.

However, if you would like more control over when exactly the logs are dumped (at the end of each epoch, or each x number of epochs, for example), you can instead force the `LossManager` to dump its buffers to the loss log with:
```python
loss_manager.update_loss_log(silent=True, capture=False)
```

> [!TIP]
> `LossManager.update_loss_log()` has two optional boolean arguments, `silent` (default=True) and `capture` (default=False). If `silent` if `False`, the `LossManager` will print a string to the console after it has dumped it values to the log with a timestamp showing how long it took to perform the operation. If silent is `False` and the other optional argument, `capture`, is `True` however, the string that would have been printed is instead returned by the function. 
## Convenience Functions
#### The `LossManager` includes two convenience functions for printing debug values to the console during training. These are:
### `LossManager.print_losses()`
> From: [nectargan.losses.loss_manager.print_losses()](/nectargan/losses/loss_manager.py)

    Prints (or returns) a string of all the most recent loss values.

    Note: This function uses the last value stored in LossManager.history 
    for each loss. As such, this function should generally be called AFTER 
    all of the registered loss functions have been run for the batch. 
    Calling it before running some or all of the loss funtions could lead 
    to unexpected results.

    By default, this function will print a string of all registered losses 
    and their most recent values, tagged with epoch and iter, formatted as:

    "(epoch: {e}, iters: {i}) Loss: {L_1_N}: {L_1_V} {L_2_N}: {L_2_V} ..."

    Key:
        e : input epoch
        i : input iter
        L_X_N : Loss X name
        L_X_V : Loss X value
### `LossManager.print_weights()`
> From: [nectargan.losses.loss_manager.print_weights()](/nectargan/losses/loss_manager.py)

    Prints (or optionally returns) loss weight information.

    Note: This function uses the last weight value stored in:
        - `LossManager.schedule.current_value`
    Since this value is updated immediately after the associated loss has
    been run, the weight values that this function prints (or returns) are 
    the ones which will be applied the next time the loss is run.

    By default, this function will print a string of all registered losses 
    and their most recent weights formatted as:

    "Loss weights: {L_1_N}: {L_1_W} {L_2_N}: {L_2_W} ..."

    Key:
        L_X_N : Loss X name
        L_X_W : Loss X weight
## Loss Specs?