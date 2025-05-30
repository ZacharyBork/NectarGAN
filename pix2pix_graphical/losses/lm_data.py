from __future__ import annotations
from dataclasses import dataclass, field
from torch import nn, Tensor, empty

from pix2pix_graphical.scheduling.data import WeightSchedule

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
        schedule : A WeightSchedule object which can be used to schedule the
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
    schedule: WeightSchedule=field(default_factory=WeightSchedule)
    last_loss_map: Tensor = field(default_factory=lambda: empty(0))
    store_history: bool=False
    history: LMHistory = field(default_factory=LMHistory)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        '''Post-init function for LMLoss.
        
        This function is used to set the starting value of self.schedule's
        current_weight to self.loss_weight if the user didn't initialize the
        WeightSchedule with their own values, and sets self.current weight to
        self.schedule.initial_weight if they did. This is done so that the
        current weights of all registered LMLoss objects can be applied in the
        same way regardless of whether scheduling is being used for the loss.

        If the user does init the scheduler when instantiating the LMLoss, this
        function will also subtract 1 from the selected start and end epoch 
        since, for convenience of logging and printing, the Trainer class 
        treats epoch as though it indexes from 1. 
        '''
        if self.schedule == WeightSchedule():
            self.schedule.current_weight = self.loss_weight
        else: 
            self.loss_weight = self.schedule.initial_weight
            self.schedule.current_weight = self.schedule.initial_weight
            self.schedule.start_epoch = self.schedule.start_epoch - 1
            self.schedule.end_epoch = self.schedule.end_epoch - 1
