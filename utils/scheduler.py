from torch import optim
from torch.optim.lr_scheduler import LambdaLR

class LRScheduler():
    def __init__(self, optimizer: optim.Adam, n_epochs: int, n_epochs_decay: int):
        '''Initialization function for the LR scheduler class.'''
        self.n_epochs = n_epochs
        self.n_epochs_decay = n_epochs_decay
        self.optimizer = optimizer

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.linear_decay_schedule)
        self._old_lr = self._new_lr = optimizer.param_groups[0]['lr']

    def linear_decay_schedule(self, epoch):
        '''Defines a linear decay schedule whereby the learning rate remains
        at a constant for LRScheduler.n_epochs, then linearly decays to 0 over
        n_epochs_decay. The first decay epoch is run at full LR, so the final
        decay epoch learning rate is not 0.0, but is instead:
    
        initial_learning_rate * (1.0 / LRScheduler.n_epochs_decay)
        '''
        # Full lr for n_epochs, then linear decay to 0 over n_epochs_decay
        if epoch < self.n_epochs: return 1.0
        else: return max(0.0, 1.0 - (epoch - self.n_epochs) / self.n_epochs_decay)

    def step(self):
        '''Step function for the LRScheduler. This function also stores
        the previous and the new epochs learning rates when it steps forward.
        '''
        self.scheduler.step()
        self._old_lr = self._new_lr
        self._new_lr = self.scheduler.get_last_lr() 

    def get_lr(self):
        '''To be called after step, returns the LR of the just completed epoch
        and the new epoch for printing or graphing purposes.
        '''
        return (self._old_lr, self._new_lr)
