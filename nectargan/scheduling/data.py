from __future__ import annotations
from dataclasses import dataclass, fields, MISSING
from typing import Literal, Callable

@dataclass
class Schedule:
    '''Stores loss weight scheduling information.

    Note on `LMLoss` / `Schedule` initialization:
        When creating an LMLoss object, if you do not initialize the values of 
        its internal `Schedule`, it's created with a default `Schedule`. This 
        schedule doesn't really do anything. If it was actually used, it would 
        start at 1.0, do nothing for 10,000,000 epochs, and end at 1.0. 
        
        In the loss manager, when we are applying loss weights, we first check 
        to see if the `LMLoss`'s `Schedule` matches the default definition 
        (i.e. the user did not set up a schedule when they initialized the 
        LMLoss), and if it does, we effectively bypass the schedule and instead 
        apply whatever weight value was passed as `loss_weight` when the LMLoss 
        was instantiated.         

    Args:
        schedule : Either the name of an included scheduling function as a 
            string, or a custom callable scheduling function. Custom functions 
            must accept a `Schedule` object and an int (current epoch), and 
            return a float. Examples can be found at:
                
                - `nectargan.scheduling.schedules`
            
            LossManager schedule function calling implementation:
                
                - `LossManager._weight_loss()`
        
        start_epoch : Epoch to start increasing or decreasing the loss values.
        end_epoch : Epoch to stop increasing or decreasing the loss values.
        initial_value : The value to use until start_epoch.
        target_value : The value to interpolate to at, and hold after, end_epoch.
        current_value : Not meant to be set directly, this value is instead 
            set by the `Schedule`'s function, and used by the LossManager to 
            weight the return value of registered losses when they are run.
    '''
    NO_ARG = object()

    schedule: (
        Literal['linear', 'exponential'] | 
        Callable[[Schedule, int], None])='linear'
    start_epoch: int=0
    end_epoch: int=int(1e07)
    initial_value: float=1.0
    target_value: float=1.0
    current_value: float=1.0

    def _check_for_defaults(self) -> bool:
        '''Used in post_init to check if all args are at their default values.
        
        Returns:
            bool : True is all arguments are at default, otherwise False.
        '''
        for f in fields(self):
            if f.default is not MISSING: default = f.default
            elif f.default_factory is not MISSING: 
                default = f.default_factory()
            else: continue
            if not getattr(self, f.name) == default: return False
        return True

    def __post_init__(self) -> None:
        '''Post-init function for `Schedule` dataclass.
        
        This function is used to set the starting value of the Schedule's
        `current_value` to its `initial_value` if the user initialized the
        Schedule with their own values (i.e. not all default).
        
        It will also subtract 1 from `start_epoch` and `end_epoch` in that case
        to compensate for the fact that the Trainer classes treat epoch as 
        though it was indexed from 1 rather than 0.
        '''
        if not self._check_for_defaults():
            self.current_value = self.initial_value
            self.start_epoch = self.start_epoch - 1
            self.end_epoch = self.end_epoch - 1

    def __eq__(self, value: Schedule) -> bool:
        '''Checks if `start_epoch` and `end_epoch` are at their default values.

        Since it's relatively safe to assume that no one is going to be 
        training a model for 10,000,000 epochs([0, 1e07]), this basically 
        serves as a check to see if the Schedule instance is being used for the 
        parent object. This frees us up to initialize weight values when the 
        parent object is first instantiated, since those are no longer checked
        when performing an __eq__ operation.

        Args:
            value : The other Schedule object which is being compared.
        
        Returns:
            bool : True if start_epoch and end_epoch match, otherwise false.
        '''
        return (
            value.start_epoch == self.start_epoch and 
            value.end_epoch == self.end_epoch)
            