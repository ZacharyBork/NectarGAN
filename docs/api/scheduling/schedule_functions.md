# NectarGAN API (Scheduling) - Schedule Functions
> [*`NectarGAN API - Home`*](../../api.md)
> [*`NectarGAN API - Scheduling`*](../scheduling.md)
#### Schedule functions allow you to define simple mathematical functions, which can then by plugged in to the [`Schedule`](../scheduling/schedule_dataclass.md) dataclass and used to drive scheduling for any parameter in your model.
Reference: [`nectargan.scheduling.schedules`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/scheduling/schedules.py)

*These are easy to use, but a little complex to understand at first, so let's start slow with...*
## What is a Schedule function?


**Just a simple Python function which itself describes a simple time-dependent mathematical function.** A schedule function has two things passed to it as input arguments any time it is evaluated:

1. **An epoch value.** This is used to define the sample point for the function.
2. **A `Schedule` object** (see [here](../scheduling/schedule_dataclass.md) for more info).

Schedule functions can be broken down into a few parts:
1. An initial value.
2. A target value.
3. A start epoch, until which the initial value will be returned.
4. An end epoch, after which the target value will be returned.

***So, up until the input epoch passed to the schedule function is equal to the start epoch, the schedule function returns the initial value. Then, in the epochs between the start and end epoch, the schedule function interpolates from the initial value to the target value. And finally, in the epochs after the end epoch, the function returns the target value.***

### Let's have a look at an example schedule to help clarify things:
> ###### From: [`nectargan.scheduling.schedules.ScheduleDefs`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/scheduling/schedules.py#L5)
> ```python
> def linear(schedule: Schedule, epoch: int) -> float:
>         '''Defines a linear loss weight schedule.
> 
>         Graph:
>         - https://www.desmos.com/calculator/xaponwctch
>         - e1, e2 : start, end epoch
>         - v1, v2 : start, end value
> 
>         Args:
>             schedule : Schedule object to use when computing the new weight.
>             epoch : Current epoch that the time this function is called.
>         '''
>         initial, target = schedule.initial_value, schedule.target_value
> 
>         # Normalized sample position from current epoch
>         sample = ((float(epoch) - float(schedule.start_epoch)) / 
>                   (float(schedule.end_epoch) - float(schedule.start_epoch)))
>         sample = max(0.0, min(1.0, sample)) # Clamp value [0.0, 1.0]
>         
>         # Sample function at that position
>         value = initial + sample * (target - initial)
> 
>         # Get largest and smallest of weight values
>         lowest = max(0.0, min(initial, target))
>         highest = max(initial, target)
> 
>         # Return the current weight value.
>         return min(highest, max(lowest, value))
> ```
> *Clickable graph link:* https://www.desmos.com/calculator/xaponwctch

**Alright, not too complicated. Looking at the function, we can see that we are:**
1. Getting our initial and target values from the input `Schedule` object.
2. Calculating a sample position, where the `sample` value is `0.0` at the `schedule.start_epoch`, then interpolates linearly to `1.0` over the time between `schedule.start_epoch` and `schedule.end_epoch`.
3. Then we clamp the `sample` value to the (0, 1) range.
4. Next, we use that sample value to sample a linear growth/decay function, calculated as `initial_value + sample * (target_value - initial_value)` (see [graph](https://www.desmos.com/calculator/xaponwctch)).
5. Get the lowest allowed value (note that the `linear` schedule function is clamped to a min of `0.0`) and the highest allowed value.
6. Return the minimum value if value < minimum, the maximum value if value > maximum, or the value itself if neither is true.

**Pretty simple, right?** The other standard decay schedule, [`exponential`](https://github.com/ZacharyBork/NectarGAN/blob/main/nectargan/scheduling/scheduler.py#L35), works in much the same way, with only a couple small differences. Since this function uses logorithmic interpolation, which can result in a `ZeroDivisionError`, it provides a couple extra variables:
| Variable | Description |
| :---: | --- |
`allow_zero_weights` | If `True`, the function will allow a value of `0.0` for initial or target weight, but will add a small epsilon value to the `0.0` value to avoid the `ZeroDivisionError`.
`epsilon` | The epsilon value to add if either initial or target weight is `0.0`.
`silent` | If `allow_zero_weights` if `False` and this value is also `False`, this funtion will raise a `ZeroDivisionError`. If `allow_zero_weights` is `False` and this is `True`, however, rather than raising an exception, it will instead return the initial value until the `schedule.end_epoch`, at which point it will instead return the target value, effectively allowing it to fail silently.

---