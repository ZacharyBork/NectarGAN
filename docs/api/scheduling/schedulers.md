# NectarGAN API (Scheduling) - Schedulers
> [*`NectarGAN API - Home`*](../../api.md)
> [*`NectarGAN API - Scheduling`*](../scheduling.md)
#### The NectarGAN API provides two classes for managing schedules, depending on the objective. These are:
## The Scheduler class
**The `Scheduler` is a wrapper class to simplify the process of interacting with [`Schedule`](../scheduling/schedule_dataclass.md) objects.**
### Creating a Scheduler instance
 A `Scheduler` can be created as follows. First, we will create a `Schedule` object:
```python
from nectargan.scheduling.data import Schedule

schedule = Schedule(
    schedule='linear',   # Define a linear decay schedule
    initial_value=100.0, # which starts at a value of 100.0,
    start_epoch=100,     # and which begins decaying at epoch 100,
    target_value=0.0,    # until finishing the decay at a value of 0.0
    end_epoch=200        # at epoch 200
)
```
> [!NOTE]
> This schedule should look familiar if you have read the [original Pix2pix paper](https://arxiv.org/pdf/1611.07004), as this `Schedule` object defines a schedule which mirrors the one they used for the learning rate in their original implementation.

**Then, we can use our `Schedule` object to initialize a `Scheduler` like so:**
```python
from nectargan.scheduling.scheduler import Scheduler

scheduler = Scheduler(schedule)
```
**Pretty simple. Behind the scenes, this will do some quick schedule validation. Then, assuming the current schedule is valid, we're ready to use it.** This is also very simple, although there is some pre-requisite knowledge which is required to better understand how we use `Schedules` in NectarGAN, so you are encouraged to quickly read through the [schedule function documentation](../scheduling/schedule_functions.md) before continuing here.

Now that we have created a `Scheduler` instance and assigned it our linear decay `Schedule`, we can evaluate our schedule function with [`Scheduler.eval_schedule()`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/scheduling/scheduler.py#L32) as follows:
```python
for epoch in range(200):
    current_value = scheduler.eval_schedule(epoch)
    print(f'Epoch: {epoch+1} - Value: {current_value}')
```
###### Result
    Epoch: 1 - Value: 1.0
    Epoch: 2 - Value: 1.0
    [...]
    Epoch: 99 - Value: 1.0
    Epoch: 100 - Value: 1.0
    Epoch: 101 - Value: 0.99
    Epoch: 102 - Value: 0.98
    Epoch: 103 - Value: 0.97
    [...]
    Epoch: 198 - Value: 0.02
    Epoch: 199 - Value: 0.01
    Epoch: 200 - Value: 0.0

## The TorchScheduler class
**The [`TorchScheduler`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/scheduling/scheduler_torch.py) is a compatibility wrapper around the native [`torch.optim.lr_scheduler`](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate), allowing you to use scheduling functions interchangeably while retaining the native scheduler's deep integration with the native optimizers.**

A `TorchScheduler` instance can be created as follows. First, we create a schedule in the same way we did for the base `Scheduler`:
```python
from nectargan.scheduling.data import Schedule

schedule = Schedule(
    schedule='linear',   # Define a linear decay schedule
    initial_value=100.0, # which starts at a value of 100.0,
    start_epoch=100,     # and which begins decaying at epoch 100,
    target_value=0.0,    # until finishing the decay at a value of 0.0
    end_epoch=200        # at epoch 200
)
```
For the `TorchScheduler`, however, we also need to pass it an optimizer. **This should be the optimizer that is linked to the network you are trying to schedule the learning rate for.** Then, we can can create a `TorchScheduler` instance like so:
```python
from torch.optim import Adam
from nectargan.scheduling.scheduler_torch import TorchScheduler
optimizer = Adam(params, initial_lr, betas) # Should be the real optimizer of the net to schedule

scheduler = TorchScheduler( # Create a TorchScheduler
    optimizer=optimizer,    # and pass it the optimizer
    schedule=schedule       # and the schedule
)
```
Then you can just call the [`TorchScheduler.step()`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/scheduling/scheduler_torch.py#L19) wrapper at the end of each epoch to update the learning rate as follows:
```python
scheduler.step() # Simple wrapper around torch.optim.lr_scheduler.LRScheduler.step()
```
### [`get_lr()`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/scheduling/scheduler_torch.py#L27) Utility
The `TorchScheduler` also provides a utility function called `get_lr`, which is intended to be called after `TorchScheduler.step()`, and will return a tuple of the previous learning rate, from before the most recent `step()` call, and the current learning rate, after it has been updated by `step()`

---