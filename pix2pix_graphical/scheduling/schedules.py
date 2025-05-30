from pix2pix_graphical.scheduling.data import WeightSchedule

class WeightSchedules:
    @staticmethod
    def linear(schedule: WeightSchedule, epoch: int) -> float:
        '''Defines a linear loss weight schedule.

        Graph:
            - https://www.desmos.com/calculator/xaponwctch
            - e1, e2 : start, end epoch
            - v1, v2 : start, end value

        Args:
            schedule : WeightSchedule to use when computing the new weight.
            epoch : Current epoch that the time this function is called.
        '''
        initial, target = schedule.initial_weight, schedule.target_weight

        # Normalized sample position from current epoch
        sample = ((float(epoch) - float(schedule.start_epoch)) / 
                  (float(schedule.end_epoch) - float(schedule.start_epoch)))
        sample = max(0.0, min(1.0, sample)) # Clamp value [0.0, 1.0]
        
        # Sample function at that position
        value = initial + sample * (target - initial)

        # Get largest and smallest of weight values
        lowest = max(0.0, min(initial, target))
        highest = max(initial, target)

        # Return the current weight value.
        return min(highest, max(lowest, value))
    
    @staticmethod
    def exponential(
        schedule: WeightSchedule, 
        epoch: int,
        epsilon: float=1e-09,
        allow_zero_weights: bool=True,
        silent: bool=False
    ) -> float:
        '''Defines a exponential loss weight schedule.

        Graph:
            - https://www.desmos.com/calculator/adqximccwt
            - e1, e2 : start, end epoch
            - v1, v2 : start, end value

        Args:
            schedule WeightSchedule to use when computing the new weight.
            epoch : Current epoch that the time this function is called.
            epsilon : Epsilon for initial and target values. Only used if
                `allow_zero_weights` is True (default).
            allow_zero_weights : If True (default), this function will allow
                initial and target learning rate to be zero, and will still use
                standard logarithmic interpolation, but any zeros values will
                have a small `epsilon` value (default 1e-09) added to them 
                before interpolation to avoid division by zero. If false, it
                will either raise an error, or bailout without interpolating
                depenging on the value of `silent`. 
            silent : See `allow_zero_weights`.

        Raises:
            ZeroDivisionError : If `schedule.initial_weight` or `schedule.
                target_weight` are zero and both `allow_zero_weights` and
                `silent` are False.
        '''
        initial, target = schedule.initial_weight, schedule.target_weight

        # Normalized sample position from current epoch
        sample = ((float(epoch) - float(schedule.start_epoch)) / 
                  (float(schedule.end_epoch) - float(schedule.start_epoch)))
        sample = max(0.0, min(1.0, sample)) # Clamp value [0.0, 1.0]
        
        if initial == 0.0 or target == 0.0:
            if allow_zero_weights: # Stops decay just above zero
                initial = max(initial, epsilon)
                target = max(target, epsilon)
            else:     # Bailout if divide by zero
                if silent: return target if sample >= 1.0 else initial
                else: # Or error if silent=False
                    message = (
                        f'Attemped division by zero during weight scheduling '
                        f'operation. Please check settings of exponential '
                        f'schedules or enable `allow_zero_weights`.')
                    raise ZeroDivisionError(message)
        
        # Sample function at that position
        value = initial * (target / initial) ** sample

        # Get largest and smallest of weight values
        lowest = max(0.0, min(initial, target))
        highest = max(initial, target)

        # Return the current weight value.
        return min(highest, max(lowest, value))

# Map of valid default schedule functions
schedule_map = { 
    'linear': WeightSchedules.linear, 
    'exponential': WeightSchedules.exponential
}