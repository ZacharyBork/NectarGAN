from torch import optim

from nectargan.scheduling.data import Schedule
from nectargan.scheduling.scheduler import Scheduler

class TorchScheduler(Scheduler):
    def __init__(
            self, 
            optimizer: optim.Optimizer, 
            schedule: Schedule=Schedule,
        ) -> None:
        '''Initialization function for the TorchScheduler class.'''
        super().__init__(schedule)

        self.scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=self.eval_schedule)
        self._old_lr = self._new_lr = optimizer.param_groups[0]['lr']

    def step(self):
        '''Step function for the LRScheduler. This function also stores
        the previous and the new epochs learning rates when it steps forward.
        '''
        self.scheduler.step()
        self._old_lr = self._new_lr
        self._new_lr = self.scheduler.get_last_lr()[0]

    def get_lr(self):
        '''To be called after step, returns the LR of the just completed epoch
        and the new epoch for printing or graphing purposes.
        '''
        return (self._old_lr, self._new_lr)
