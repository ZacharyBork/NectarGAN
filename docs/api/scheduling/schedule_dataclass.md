# NectarGAN API (Scheduling) - Schedule Dataclass
> [*`NectarGAN API - Home`*](../../api.md)
> [*`NectarGAN API - Scheduling`*](../scheduling.md)
#### In the NectarGAN scheduling system, schedules are defined by `Schedule` objects, small drop-in dataclass instances which define everything about a given schedule, and which can be fed to either `Scheduler` class to define scheduling for any parameter.
Reference: [`nectargan.scheduling.data`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/scheduling/data.py)
## Variables
| Variable | Description |
| :---: | --- |
`schedule` | Either the name of a built in scheduling function as a string, or a custom [schedule function](../scheduling/schedule_functions.md).
`start_epoch` | Epoch to start increasing or decreasing the loss values.
`end_epoch` | Epoch to stop increasing or decreasing the loss values.
`initial_value` | The value to use until start_epoch.
`target_value` | The value to interpolate to at, and hold after, end_epoch.
`current_value` | Not meant to be set directly, this value is instead set by the `Schedule`'s [`__post_init__`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/scheduling/data.py#L67) function. It is primarily used by the [`LossManager`](../losses/lossmanager.md) to weight the return value of registered losses when they are run.
> [!NOTE]
> When a `Schedule` instance is first created, it the default values are left unchanged, it will be created with a schedule which doesn't really do anything. Were the default schedule actually used, it would start at 1.0, do nothing for 10,000,000 epochs, and end at 1.0.
### Creating a `Schedule` instance
The following is an example of how you can create a `Schedule` instance using one of the default [schedule functions](../scheduling/schedule_functions.md), [`linear`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/scheduling/schedules.py#L5):
```python
from nectargan.scheduling.data import Schedule

schedule = Schedule(
    schedule='linear',
    initial_value=100.0,
    start_epoch=100,
    target_value=0.0,
    end_epoch=200 
)
```
This will create a schedule whereby the value is held constant at `100.0` for the first 100 epochs. Then, over the next hundred epochs (i.e. epoch 100-200), it will decay linearly to the target value of `0.0`.

**It is also possible to define your own schedule functions (see [here](../scheduling/schedule_functions.md) for more info).** A `Schedule` object can be initialized with a custom schedule function like so. First, we will create our schedule function:
```python
from nectargan.scheduling.data import Schedule

def my_schedule(schedule: Schedule, epoch: int) -> float:
    # Define schedule
    return output

schedule = Schedule(
    schedule=my_schedule,
    initial_value=100.0,
    start_epoch=100,
    target_value=0.0,
    end_epoch=200 
)
```
**Here, we define the `Schedule` the same way we did previously, except, rather than passing it the name of one of the [default schedules](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/scheduling/schedules.py#L99), we instead directly pass it a reference to the `Callable` which defines our schedule.**
## Methods
### [`Schedule._check_for_defaults()`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/scheduling/data.py#L53)
    Used in post_init to check if all args are at their default values.
        
    Returns:
        bool : True is all arguments are at default, otherwise False.
### [`Schedule.__post_init__()`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/scheduling/data.py#L67)
    Post-init function for `Schedule` dataclass.
        
    This function is used to set the starting value of the Schedule's
    `current_value` to its `initial_value` if the user initialized the
    Schedule with their own values (i.e. not all default).
    
    It will also subtract 1 from `start_epoch` and `end_epoch` in that case
    to compensate for the fact that the Trainer classes treat epoch as 
    though it was indexed from 1 rather than 0.

### [`Schedule.__eq__()`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/scheduling/data.py#L83)
    Checks if `start_epoch` and `end_epoch` are at their default values.

    Since it's relatively safe to assume that no one is going to be 
    training a model for 10,000,000 epochs([0, 1e07]), this basically 
    serves as a check to see if the Schedule instance is being used for the 
    parent object. This frees us up to initialize weight values when the 
    parent object is first instantiated, since those are no longer checked
    when performing an __eq__ operation.

    Args:
        value : The other Schedule object which is being compared.
    
    Returns:
        bool : True if start_epoch and end_epoch match, otherwise false.

---