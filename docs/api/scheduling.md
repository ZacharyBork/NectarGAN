# NectarGAN API - Scheduling
> [*`NectarGAN API - Home`*](/docs/api.md)
#### The NectarGAN API provides a modular, highly configurable solution to managing scheduling for any parameter in your models.
## Scheduling dataclass
The core of the NectarGAN scheduling system is the [`Schedule`](/nectargan/scheduling/data.py) dataclass, an easy to use tool which (along with the [scheduling function system](/docs/api/scheduling/schedule_functions.md)) allows you to quickly define complex schedules which can be applied to any paramaeter in your model with relative ease.

**Please see [here](/docs/api/scheduling/schedule_dataclass.md) for more information.**
## Schedule Functions
The NectarGAN scheduling system works via a drop-in schedule function system. This system allows you to define reusable schedule functions, and easily drop them in to anything you'd like to schedule.

**Please see [here](/docs/api/scheduling/schedule_functions.md) for more information.**
## Scheduler Classes
Scheduling in the NectarGAN API is managed by two main classes: a general scheduler, which can be used for basically anything, and a compatibility wrapper around PyTorch's native [`torch.optim.lr_scheduler`](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate), allowing you to use scheduling functions interchangeably while retaining the native scheduler's deep integration with the native optimizers.

**Please see [here](/docs/api/scheduling/schedulers.md) for more information.**
## Scheduler Integrations
`Schedules` are deeply integrated into other core components of the NectarGAN API.

**Please see [here](/docs/api/scheduling/integrations.md) for more information.**

---