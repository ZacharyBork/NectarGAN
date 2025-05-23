from torch import optim
from torch.optim.lr_scheduler import LambdaLR

class LRScheduler():
    def __init__(self, optimizer: optim.Adam, n_epochs: int, n_epochs_decay: int):
        self.n_epochs = n_epochs
        self.n_epochs_decay = n_epochs_decay
        self.optimizer = optimizer

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.linear_decay_schedule)
        self._old_lr = self._new_lr = optimizer.param_groups[0]['lr']

    def linear_decay_schedule(self, epoch):
        # Full lr for n_epochs, then linear decay to 0 over n_epochs_decay
        if epoch < self.n_epochs: return 1.0
        else: return max(0.0, 1.0 - (epoch - self.n_epochs) / self.n_epochs_decay)

    def step(self):
        self.scheduler.step()
        self._old_lr = self._new_lr
        self._new_lr = self.scheduler.get_last_lr() 

    def get_lr(self):
        return (self._old_lr, self._new_lr)
