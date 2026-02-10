import torch

class MemoryInfo_CUDA():
    '''Helper class for querying and managing info about GPU memory usage.'''
    def __init__(self) -> None:
        self.peak = 0.0
        self.avg_peak = 0.0
        self.allocated = 0.0
        self.reserved = 0.0

        self.update_counter = 0
        self._total = 0

    def _reset_update_counter(self) -> None:
        self.update_counter = 0

    def _get_current_usage(self) -> None:
        self.allocated = torch.cuda.memory_allocated() / 1024**2
        self.reserved = torch.cuda.memory_reserved() / 1024**2

    def _reset_peak(self) -> None:
        torch.cuda.reset_peak_memory_stats()
        self.peak = 0.0

    def _update_peak_usage(self) -> None:
        self._reset_peak()
        self.peak = torch.cuda.max_memory_allocated() / 1024**2

    def _update_avg_peak(self) -> None:
        self.update_counter += 1
        self._total += self.peak
        self.avg_peak = self._total / self.update_counter 

    def reset_avg_peak(self) -> None:
        self.avg_peak = 0.0

    def get_summary(self) -> str:
        return torch.cuda.memory_summary()
    
    def print_summary(self) -> None:
        print(self.get_summary())

    def print_current_usage(self) -> None:
        print(f'GPU memory allocated: {self.allocated}')
        print(f'GPU memory reserved: {self.reserved}')

    def print_peak_usage(self) -> None:
        print(f'Peak GPU memory usage: {self.peak}')

    def update(self) -> None:
        self._update_peak_usage()
        self._update_avg_peak()





