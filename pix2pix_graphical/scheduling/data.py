from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Callable

@dataclass
class WeightSchedule:
    '''Helper dataclass to store loss weight scheduling information.

    Note on LMLoss / WeightSchedule initialization:
        When an LMLoss object is initialized, it is created with a factory
        default WeightSchedule. This schedule doesn't really do anything. If it 
        was actually used, it would start and 1.0, do nothing for 10,000,000 
        epochs, and end at 1.0. 
        
        In the loss manager, when we are applying loss weights, we first check 
        to see if the LMLoss's WeightSchedule matches the default definition 
        (i.e. the user did not set up a schedule when they initialized the 
        LMLoss), and if it is, we bypass the schedule and instead apply 
        whatever weight value was passed as `loss_weight` when the LMLoss was 
        instantiated.

    Args:
        schedule : Either the name of a pre-built decay function as a string,
            or a custom callable decay function. Custom functions must accept
            an LMWeightsSchedule object and an int for the current epoch, these
            will both be passed to the function by the LossManager when the
            parent loss is called so you can access the data in this dataclass
            directly from the loss function, and must return a float (the new
            loss weight value). Examples can be found at:
                
                - `pix2pix_graphical.losses.scheduling`
            
            The schedule function calling implementation can be found here:
                
                - `LossManager._weight_loss()`
        
        start_epoch : Epoch to start increasing or decreasing the loss values.
        end_epoch : Epoch to stop increasing or decreasing the loss values.
        initial_weight : Loss weight value to use until start_epoch.
        target_weight : Weight to interpolate to at, and hold after, end_epoch.
        current_weight : Not meant to be set directly, this value is instead 
            set by the WeightSchedules functions, and used by the LossManager 
            to weight the return value of registered losses when they are run.
    '''
    schedule: (
        Literal['linear', 'exponential'] | 
        Callable[[WeightSchedule, int], None])='linear'
    start_epoch: int=0
    end_epoch: int=int(1e07)
    initial_weight: float=1.0
    target_weight: float=1.0
    current_weight: float=1.0

    def __eq__(self, value: WeightSchedule) -> bool:
        '''Checks see if start_epoch and end_epoch are at their default values.

        Since it's relatively safe to assume that no one is going to be 
        training a model for 10,000,000 epochs, this basically serves as a 
        check to see if the WeightSchedule instance is being used for the 
        parent LMLoss and frees us up to initialize weight values when the 
        parent object is first instantiated, since those are no longer checked
        when performing an __eq__ operation.
        
        Returns:
            bool : True if schedule is default, otherwise false.
        '''
        return (
            value.start_epoch == self.start_epoch and
            value.end_epoch == self.end_epoch)