from pix2pix_graphical.losses.lm_data import LMWeightSchedule

class WeightSchedules:
    @staticmethod
    def linear(schedule: LMWeightSchedule, epoch: int) -> float:
        '''Defines a linear loss weight schedule.

        Graph:
            - https://www.desmos.com/calculator/xaponwctch
            - e1, e2 : start, end epoch
            - v1, v2 : start, end value

        Args:
            schedule LMWeightSchedule to use when computing the new weight.
            epoch : Current epoch that the time this function is called.
        '''
        initial, target = schedule.initial_weight, schedule.target_weight

        # Get largest and smallest of weight values
        lowest = max(0.0, min(initial, target))
        highest = max(initial, target)

        # Normalized sample position from current epoch
        sample = ((float(epoch) - float(schedule.start_epoch)) / 
                  (float(schedule.end_epoch) - float(schedule.start_epoch)))
        
        # Sample function at that position
        value = initial + sample * (target - initial)

        # Return the current weight value.
        return min(highest, max(lowest, value))
