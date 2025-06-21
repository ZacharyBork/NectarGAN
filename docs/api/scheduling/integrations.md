# NectarGAN API (Scheduling) - Integrations
> [*`NectarGAN API - Home`*](/docs/api.md)
> [*`NectarGAN API - Scheduling`*](/docs/api/scheduling.md)
#### The `Schedule` system is deeply integrated with other core components of the NectarGAN API. The ways in which it interacts with those components are detailed here.

> [!NOTE]
> Currently, the main `Schedule` integration is with the [`LossManager`](/docs/api/losses/lossmanager.md). This document has been left open-ended though, so that it can be expanded in the future as more integrations are built out.

> [!WARNING]
> **It is strongly encouraged to read the other scheduling related documentation before this one. While an effort will be made in this document to thoroughly explain all relevant concepts, much of it would be outside the scope of this document alone, and many concepts discussed here may be confusing without the prior context which those documents offer.**
>
> **It is also encouraged to read the documentation relevant to the API component which is being discussing, which will be linked at the top of each section.**

## Schedules and the LossManager
Reference [`nectargan.losses.loss_manager.LossManager`](/nectargan/losses/loss_manager.py) [`/docs/api/losses/lossmanager`](/docs/api/losses/lossmanager.md)

**The `LossManager` is deeply integrated with the `Schedule` system, allowing you to easily assign independent weight schedules to any loss registered with a given `LossManager` instance.**
> [!NOTE]
> *Peeling back the curtain just a little bit, I didn't originally intend for the scheduling system to have the level of complexity that it ended up having. Originally, the [`Scheduler`](/docs/api/scheduling/schedulers.md) was just intended to be a simple helper class for the loss manager to allow for scheduling of loss weights, since PyTorch already offers a robust and simple to use learning rate scheduler ([`torch.optim.lr_scheduler.LRScheduler`](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LRScheduler.html#lrscheduler)).*
>
> *However, then I built it, and I thought it would be cool to be able to use drop-in [functions to define the schedules](/docs/api/scheduling/schedule_functions.md). Then once that was working, I started thinking it might also be cool to be able to use those schedule functions to drive the native PyToch `LRScheduler`. Things just kinda kept growing, and that's how we ended up here. There are still some mentions of the `LossManager` and its related components in docstrings and comments and whatnot in the scheduling components, though, and this is why.*

### To understand how this interaction works, we first need to break it down in to parts.

**Let's start by having a quick look as the [`LMLoss`](/nectargan/losses/lm_data.py#L26)**. Looking through the member variables, we can see that one of them is a `Schedule` object called `schedule`. We can also see that at init, the `schedule` variable, if it is not overridden by the declaration, is assigned a default factory `Schedule`.

**Next, let's have a look at the `LMLoss`'s [`__post_init__`](/nectargan/losses/lm_data.py#L62) method:**
```python
def __post_init__(self) -> None:
    '''Post-init function for `LMLoss`.
    
    This function will:
        A.) Set the starting value of `self.schedule`'s `current_value` to 
            `self.loss_weight` if the user didn't initialize the `Schedule` 
            with their own values.

            --------------------------- Or: ---------------------------
            
        B.) Set `self.current_weight` to `self.schedule.initial_value` if 
            they did.
            
    This is done so that the current weights of all registered `LMLoss` 
    objects can be applied in the same way regardless of whether scheduling 
    is being used for the loss. 
    '''
    if self.schedule == Schedule():
        self.schedule.current_value = self.loss_weight
    else: self.loss_weight = self.schedule.initial_value   
```
### Okay, pretty simple in theory. Let's see how it functions, though.
To do that, we will head over `LossManager`, and look for the member function called `_weight_loss()`, which is called during [`LossManager.compute_loss_xy()`](/nectargan/losses/loss_manager.py#L639), just after the given loss function was run, to apply weighting to the loss result before it is returned:
> ###### From [`nectargan.losses.loss_manager.LossManager`](/nectargan/losses/loss_manager.py#L606)
> ```python
> def _weight_loss(
>         self, 
>         loss_entry: LMLoss,
>         loss_value: torch.Tensor,
>         epoch: int,
>         **weight_kwargs: Any
>     ) -> torch.Tensor:
>     '''Applies weighting to a loss value from an LMLoss object definition.
> 
>     Args:
>         loss_entry : The LMLoss object which defines the weighting.
>         loss_value : The loss value to apply the weighting to.
>         epoch : The epoch that the loss value was calculated during.
>     '''
>     s = loss_entry.schedule # Get LMLossSchedule
>     # If the loss isn't scheduled, just apply weight and return
>     if s == Schedule(): return loss_value * s.current_value
>     
>     # Otherwise get and apply currently scheduled weights
>     fn = s.schedule # Schedule function definition
>     if isinstance(fn, Callable): s.current_value = fn(s, epoch)
>     elif isinstance(fn, str) and fn in schedule_map.keys():
>         s.current_value = schedule_map[fn](s, epoch, **weight_kwargs)
>     else: 
>         message = (
>             f'Invalid schedule type: {type(fn)}: ({fn})\n'
>             f'Valid types are: Literal["linear"] | '
>             f'Callable[[Schedule, int], None]')
>         raise TypeError(message)
>     return loss_value * s.current_value
> ```
**Alright so looking at the function, we can see it takes as input:**
- An `LMLoss` object, the current loss which is being run from a call to `LossManager.compute_loss_xy()`.
- A `torch.Tensor` representing the returned result of the given loss function to apply weighting to.
- An `integer` representing the current epoch at the time the loss is called. This is used to sample the scheduling function (please see [here](/docs/api/losses/loss_functions.md) for more information).
- Optional kwargs for the schedule function, passed to `compute_loss_xy` as a dict of `loss_kwargs`. The `weight_kwargs` dict is extracted from `loss_kwargs`, unpacked, and passed to `_weight_losses`.

**Let's now walk through step by step to see what exactly it's doing:**
1. First, we extract the `Schedule` from the input `LMLoss`.
2. We then do a check to see if the `Schedule` is a default `Schedule` (i.e. the user did not pass a custom schedule as an argument when creating the given `LMLoss`).
    - If is is a default schedule, we just get the current value from the schedule (because remember, looking back at the `LMLoss`'s `__post_init__` method, if the `Schedule` that the `LMLoss` was initialized with was a default `Schedule`, we simply assign the `LMLoss`'s `loss_weight` to the `Schedule`'s `current_value`), multiply the input `loss_value` by this current value, and return it.
    - If the `LMLoss` does not have a default schedule, however, we instead:
3. Retrieve the [schedule function](/docs/api/scheduling/schedule_functions.md) from the `Schedule`.
4. Then we check to see if the `Schedule`'s schedule definition is a `Callable` (i.e. the `Schedule` was created with a custom schedule function)
    - If it is a `Callable`, we just run the function directory, passing it the `Schedule` object reference, the current epoch value, and any `weight_kwargs` which were passed through, and assigning the return value to the `Schedule`'s `current_weight`.
    - Otherwise we:
5. Check if the schedule definition is a `str` (i.e. one of the default schedules (`linear`, `exponential`)):
    - If it is, we also check to make sure it is a valid key based on the [`schedule_map`](/nectargan/scheduling/schedules.py#L99). If the key is valid, we run the related function and assign the return value to the `Schedule`'s `current_weight`. 
    - If it's invalid, we raise a `KeyError`.
6. Finally, we take the input `loss_value`, multiply it by the `Schedule`'s `current_weight` which we just calculated via one of the above methods, and return the result.

**So, based on this, we can see that the `Schedule` is pretty integral to how the `LossManager` actually manages the losses that are registered with it.** Regardless of whether a given loss is scheduled, the LossManager uses the internal `Schedule` object of each `LMLoss` registered with it to track and apply weight values every time the given loss function is called.

### In conclusion, let's quickly look at how we can create an `LMLoss` with a weight schedule:
An `LMLoss` object can be created with a `Schedule` as shown here:
```python
import torch.nn as nn
from nectargan.losses.lm_data import LMLoss
from nectargan.scheduling.data import Schedule

L1 = nn.L1Loss()

schedule = Schedule(
    schedule='linear',   # Linear decay
    start_epoch=100,     # Starting at epoch 100
    end_epoch=200,       # And ending at epoch 200
    initial_value=100.0, # With an initial value of 100.0
    target_value=0.0     # And a target value of 0.0
)

my_loss = LMLoss(        # Create LMLoss
    name='my_loss',      # With name 'my_loss'
    function=L1,         # 'my_loss' is native PyTorch L1
    schedule=schedule,   # Assign our weight schedule
    tags=['my_tag']      # And, optionally, lookup tags
)
```

An `LMLoss` can also be directly registered with the `LossManager` with a weight schedule as follows:
```python
import torch.nn as nn
from nectargan.losses.loss_manager import LossManager
from nectargan.config.config_manager import ConfigManager
from nectargan.scheduling.data import Schedule

L1 = nn.L1Loss()

# Dummy config for LossManager
config_manager = ConfigManager('/path/to/config.json')   
loss_mananger = LossManager( # Create LossManager                             
    config=config_manager.data,
    experiment_directory='/path/to/experiment/directory')

schedule = Schedule( # Define Schedule
    schedule='linear',
    start_epoch=100,
    end_epoch=200,
    initial_value=100.0,
    target_value=0.0)

# Then register the loss with the LossManager
loss_mananger.register_loss_fn(
    name='my_loss',
    loss_fn=L1,
    schedule=schedule,
    tags=['my_tag'])
```

---