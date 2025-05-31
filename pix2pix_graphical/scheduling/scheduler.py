from typing import Callable

from torch import optim

from pix2pix_graphical.scheduling.data import Schedule
from pix2pix_graphical.scheduling.schedules import schedule_map

class Scheduler():
    def __init__(
            self, 
            optimizer: optim.Optimizer, 
            schedule: Schedule,
    ) -> None:
        '''Initialization function for the Scheduler class.'''
        self.optimizer = optimizer
        self.schedule = schedule
        self.max_value = max(schedule.initial_value, schedule.target_value)
        self.schedule_fn = self._init_schedule_fn()

    def _normalize(self, value: float) -> float:
        return value / self.max_value
    
    def _init_schedule_fn(self) -> Callable[[Schedule, int], float]:
        '''Gets and validates the current `Schedule`'s scheduling function.'''
        s = self.schedule # Get current `Schedule`
        # Otherwise get current scheduled learning rate
        fn = s.schedule # Schedule function definition
        if isinstance(fn, Callable): schedule_fn = fn
        elif isinstance(fn, str) and fn in schedule_map.keys():
            schedule_fn = schedule_map[fn]
        else: 
            message = (
                f'Invalid schedule type: {type(fn)}: ({fn})\n'
                f'Valid types are: Literal["linear"] | '
                f'Callable[[Schedule, int], None]')
            raise TypeError(message)
        return schedule_fn

    def eval_schedule(self, epoch: int) -> float:
        s = self.schedule # Get current `Schedule`
        # If no LR schedule, just apply weight and return
        if s == Schedule(): return s.current_value

        s.current_value = self.schedule_fn(s, epoch)
        assert isinstance(s.current_value, float)
        return self._normalize(s.current_value)
        
    
