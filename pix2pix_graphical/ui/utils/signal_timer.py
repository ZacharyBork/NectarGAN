import time
from typing import Literal

class SignalTimer():
    '''Basically a lap timer, but for keeping track of worker signal times.'''
    def __init__(self):
        self._total: float = 0.0 # Stores total time for average
        self._count: int = 0     # Store count of times for average

        self.previous: float | None = None # Last registered time
        self.elapsed:  float | None = None # Last elapsed time
        self.slowest:  float | None = None # Slowest recorded time
        self.fastest:  float | None = None # Fastest recorded time
        self.average:  float | None = None # Average of all recorded times

    def set_time(self) -> None:
        current = time.monotonic()
        if not self.previous is None:
            self._count += 1
            self.elapsed = current - self.previous
            self._total += self.elapsed
            self.average = self._total / self._count
            if self.slowest is None or self.elapsed > self.slowest: 
                self.slowest = self.elapsed
            if self.fastest is None or self.elapsed < self.fastest: 
                self.fastest = self.elapsed
        self.previous = current

    def get_time(
            self, 
            _type: Literal['total', 'elapsed', 'slowest', 'fastest', 'average']
        ) -> str:
        match _type:
            case 'total':   seconds = self._total
            case 'elapsed': seconds = self.elapsed
            case 'slowest': seconds = self.slowest
            case 'fastest': seconds = self.fastest
            case 'average': seconds = self.average
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f'{hours:02d}:{minutes:02d}:{secs:06.3f}'