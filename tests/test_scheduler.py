import pytest

from nectargan.scheduling.data import Schedule
from nectargan.scheduling.scheduler import Scheduler

def test_scheduler() -> None:
    '''Tests the Scheduler class functionality.

    Builds a Scheduler and runs two schedules, linear and exponential, for 10 
    epochs. Then compares the results against the expected ground truth to
    ensure the Scheduler is functioning as expected.
    '''
    expected = [
        # Linear
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        # Exponential
        [0.12589, 0.01584, 0.00199, 0.00025, 3.16227e-05, 
         3.98107e-06, 5.01187e-07, 6.30957e-08, 7.94328e-09, 0.0]]

    for index, schedule_type in enumerate(['linear', 'exponential']):
        scheduler = Scheduler(
            Schedule(schedule=schedule_type, end_epoch=10, target_value=0.0))
        result = [scheduler.eval_schedule(i) for i in range(10)]
        assert result == pytest.approx(expected[index], rel=1e-5, abs=1e-5)
