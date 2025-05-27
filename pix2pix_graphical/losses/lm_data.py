from dataclasses import dataclass, field
import torch.nn as nn
from torch import Tensor

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
    loss_weight: float | None
    last_loss_map: Tensor
    history: LMHistory
    tags: list[str]