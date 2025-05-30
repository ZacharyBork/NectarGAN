from __future__ import annotations
from typing import Literal, Callable
from dataclasses import dataclass, field
from torch import nn, Tensor, empty

@dataclass
class LMWeightSchedule:
    '''Helper dataclass to store loss weight scheduling information.

    Note on LMLoss / LMWeightSchedule initialization:
        When an LMLoss object is initialized, it is created with a factory
        default LMWeightSchedule. This schedule doesn't really do anything. If
        it was actually used, it would start and 1.0, do nothing for 10,000,000 
        epochs, and end at 1.0. 
        
        In the loss manager, when we are applying loss weights, we first check 
        to see if the LMLoss's LMWeightSchedule matches the default definition 
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
        
        type : What type of schedule to run. "growth" if the loss weight should
            increase over time, "decay" if it should decrease.
        start_epoch : Epoch to start increasing or decreasing the loss values.
        end_epoch : Epoch to stop increasing or decreasing the loss values.
        initial_weight : Loss weight value to use until start_epoch.
        target_weight : Weight to interpolate to at, and hold after, end_epoch.
        current_weight : Not meant to be set directly, this value is instead 
            set by the WeightSchedules functions, and used by the LossManager 
            to weight the return value of registered losses when they are run.
    '''
    schedule: (
        Literal['linear'] | 
        Callable[[LMWeightSchedule, int], None])='linear'
    type: Literal['growth', 'decay']='decay'
    start_epoch: int=0
    end_epoch: int=int(1e07)
    initial_weight: float=1.0
    target_weight: float=1.0
    current_weight: float=1.0

    def __eq__(self, value: LMWeightSchedule) -> bool:
        '''Checks see if start_epoch and end_epoch are at their default values.

        Since it's relatively safe to assume that no one is going to be 
        training a model for 10,000,000 epochs, this basically serves as a 
        check to see if the LMWeightSchedule instance is being used for the 
        parent LMLoss and frees us up to initialize weight values when the 
        parent object is first instantiated, since those are no longer checked
        when performing an __eq__ operation.
        
        Returns:
            bool : True if schedule is default, otherwise false.
        '''
        return (
            value.start_epoch == self.start_epoch and
            value.end_epoch == self.end_epoch)
    
@dataclass
class LMHistory:
    '''Helper dataclass to store loss value histories.

    Variables:
        losses : This is a dict[float32] to store previous loss values. After
            a loss function is run, we find the mean loss value of the return
            tensor and append it to it's corresponding LMHistory's losses list.
        weights : This is a dict[float32] to store previous loss weights. When 
            we store the loss value, we also store the value of the loss's
            weight in this dict (if it has an associated lambda value). This is 
            useful for tracking loss weights during fine tuning, or when 
            scheduling the weights for a particular loss.
    '''
    losses: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)

@dataclass
class LMLoss:
    '''Defines a registered loss function for the LossManager.

    Variables:
        name : name of the LMLoss container. Used for primarily for querying 
            registered losses. Must be unique amongst all other losses 
            registered with the LossManager instance. To get losses currently
            registered with a loss manager instance, please see:

                - LossManager.get_registered_losses()

        function : The actual loss function itself. This can be almost any
            nn.Module as long as it has a forward function that returns a
            torch.Tensor.
        loss_weight : The lambda value of the loss described by this object.
            This value defaults to 1.0 when registering a new loss, but can be 
            overwritten by passing a custom value to the loss_weight argument 
            when calling LossManager.register_loss_fn().
        schedule : An LMWeightSchedule object which can be used to schedule the
            weights of the loss during training.
        last_loss_map : This value stores resulting torch.Tensor from the last 
            time the loss function was run. It is first detached and moved
            to the CPU to reduce memory overhead.
        tag : These are user-assignable tags for the loss value, used to query 
            the loss objects in various forms. For the losses created via 
            LossManager.init_from_config(), this is used to discern whether the 
            loss result is applied to the generator or the discriminator.
    '''
    name: str
    function: nn.Module
    loss_weight: float=1.0
    schedule: LMWeightSchedule=field(default_factory=LMWeightSchedule)
    last_loss_map: Tensor = field(default_factory=lambda: empty(0))
    history: LMHistory = field(default_factory=LMHistory)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        '''Post-init function for LMLoss.
        
        This function is used to set the starting value of self.schedule's
        current_weight to self.loss_weight if the user didn't initialize the
        LMWeightSchedule with their own values, and sets self.current weight to
        self.schedule.initial_weight if they did. This is done so that the
        current weights of all registered LMLoss objects can be applied in the
        same way regardless of whether scheduling is being used for the loss.

        If the user does init the scheduler when instantiating the LMLoss, this
        function will also subtract 1 from the selected start and end epoch 
        since, for convenience of logging and printing, the Trainer class 
        treats epoch as though it indexes from 1. 
        '''
        if self.schedule == LMWeightSchedule():
            self.schedule.current_weight = self.loss_weight
        else: 
            self.loss_weight = self.schedule.initial_weight
            self.schedule.current_weight = self.schedule.initial_weight
            self.schedule.start_epoch = self.schedule.start_epoch - 1
            self.schedule.end_epoch = self.schedule.end_epoch - 1
