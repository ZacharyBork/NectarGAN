import torch
import torch.nn as nn
from typing import Any, Literal
from dataclasses import dataclass

from pix2pix_graphical.losses.losses import SobelLoss, LaplacianLoss
from pix2pix_graphical.config.config_data import Config

@dataclass
class LMLossContainer:
    '''Defines a registered loss function for the LossManager.
    
    The variables of a loss object are:
        function : The actual loss function itself. This can be almost any
            nn.Module as long as it has a forward function that returns a
            torch.Tensor.
        prev_loss : This value stores resulting torch.Tensor from the last
            time the loss function was run. It is first detached and moved
            to the CPU to reduce memory overhead.
        tag : These are user-assignable tags for the loss value, used to
            query the loss objects in various forms. For the losses created
            via LossManager.init_from_config(), this is used to discern 
            whether the loss result is applied to the generator or the 
            discriminator.
    '''
    function: nn.Module
    prev_loss: torch.Tensor
    tags: list[str]

class LossManager():
    def __init__(
            self, config: Config, 
            spec: Literal['pix2pix'] | None=None,
            sub_spec: Literal['basic', 'extended']='basic'
        ) -> None:
        self.config = config
        self.device = self.config.common.device

        # Stores loss functions and the most recent output tensor of each loss.
        # Tensor is detached and moved to CPU after corresponding Module is run.
        self.loss_fns: dict[str, LMLossContainer] = {}
        
        # Stores loss history as dict[str, float32]
        self.history: dict[str, float] = {}
        
        self.make_dummy_shape()
        self.init_from_spec(spec, sub_spec)
        
    def make_dummy_shape(self) -> None:
        '''Creates a dummy tensor of the correct model input shape.
        '''
        channels = self.config.common.input_nc  # Get input channels
        size = self.config.dataloader.crop_size # Get input W, H
        self.dummy = torch.zeros((1, channels, size, size)).to(self.device)

    def build_history(self) -> None:
        '''Builds a dict from registered losses to store loss value histories.

        For each loss registered with the LossManager, this function will add an
        entry to the LossManager.history member variable. The key will be the same
        as the lookup key for the corresponding loss function  (e.g. "G_GAN", "D_real"),
        and the value will just be an empty array to append the resulting values of 
        the associate loss function to during training.
        '''
        self.history = {}           # Clear self.history
        for loss in self.loss_fns:  # Append loss entries
            self.history[loss] = [] # 1 per registered loss
        
    def init_from_spec(
            self, 
            spec: Literal['core', 'pix2pix'], 
            sub_spec: Literal['basic', 'extended']='basic'
        ) -> None:
        '''Initialize loss manager losses from a given specification.

        This is a helper function to allow you to quickly initialize basic loss
        schemas from common specifications. If spec='core' (default if you init
        the LossManager without passing it an init_spec argument), it will just
        create an empty loss structure which you can then register your own 
        losses to.

        Args:
            spec : Specification with which to initialize the LossManager losses.
        '''
        match spec:
            case None: self.loss_fns = {}
            case 'pix2pix': 
                self.register_pix2pix_losses(mode=sub_spec)
                self.build_history()
            case _: raise ValueError(f'Invalid LossManager init_spec: {spec}')

    def register_pix2pix_losses(self, mode: Literal['basic', 'extended']='basic') -> None:
        '''Initializes loss values for pix2pix model.
        '''
        # Creates sanitized dummy tensors for loss structure
        dummy_tensor = lambda : self.dummy.clone().detach().cpu()
        BCE = nn.BCEWithLogitsLoss().to(self.device) # For G_GAN, D_real, D_fake
        L1 = nn.L1Loss().to(self.device)             # Generator L1 loss
        self.loss_fns = {
            'G_GAN': LMLossContainer(
                function=BCE, prev_loss=dummy_tensor, tags=('G')),
            'G_L1': LMLossContainer(
                function=L1, prev_loss=dummy_tensor, tags=('G')),
            'D_real': LMLossContainer(
                function=BCE, prev_loss=dummy_tensor, tags=('D')),
            'D_fake': LMLossContainer(
                function=BCE, prev_loss=dummy_tensor, tags=('D')),
        }
        if mode == 'extended':
            SOBEL = SobelLoss().to(self.device)
            self.loss_fns['G_SOBEL'] = LMLossContainer(
                function=SOBEL, prev_loss=dummy_tensor, tags=('G'))
            LAP = LaplacianLoss().to(self.device) 
            self.loss_fns['G_LAP'] = LMLossContainer(
                function=LAP, prev_loss=dummy_tensor, tags=('G'))
        self.build_history()
    
    def get_registered_losses(
            self, query: list[str] | None=None
        ) -> dict[str, LMLossContainer]:
        '''Returns a dictionary of losses registered with the LossManager.

        This function returns a dict of raw LMLossContainers which are currently 
        registered with the LossManager instance, and which have a tag that 
        matches any single item in the input query. The output dictionary keys 
        correspond to the internal names of the loss object, set when registering 
        a new loss with the LossManager. If query=None (default), this function 
        will return a dict of every LMLossContainer that is currently registered.

        Args:
            query : List of tags to search for when requesting losses. Only
                registered losses which have a tag matching the query will be
                returned.

        Returns:
            dict : The requested losses.
        '''
        output = {}
        for loss_name, loss_container in self.loss_fns.items():
            tags = getattr(loss_container, 'tags', [])
            if query is None or any(tag in tags for tag in query):
                output[loss_name] = loss_container
        return output
            
    def get_loss_values(self, query: list[str] | None=None, precision: int=2) -> dict[str, float]:
        '''Returns all of the most recent loss values (Tensor.mean()) as a dict.

        Note: This function uses the last value stored in LossManager.history for each loss.
        As such, this function should generally be called AFTER all of the registered loss
        functions have been run for the batch. Calling it before running some or all of the
        loss funtions could lead to unexpected results.

        Args:
            query : List of tags to search for when requesting losses. Only
                the values of registered losses which have a tag matching the 
                query will be returned.
            precision : Rounding precision of the returned loss values.
        Returns:
            dict : The requested loss values, mapped by their loss name.
        '''
        return {name: round(self.history[name][-1], precision)
            for name in self.get_registered_losses(query)}
    
    def get_loss_tensors(self, query: list[str] | None=None) -> dict[str, torch.Tensor]:
        '''Returns the queried LMLossContainer's prev_loss tensors.

        Note: This function uses the last value stored in LMLossContainer.prev_loss. As such, 
        this function should generally be called AFTER all of the registered loss functions 
        have been run for the batch. Calling it before running some or all of the loss funtions 
        could lead to unexpected results.

        Args:
            query : List of tags to search for. Only the tensors of registered 
                losses which have a tag matching the query will be returned.
        Returns:
            dict : The requested loss tensors, mapped by their loss name.
        '''
        return {name: container.prev_loss
            for name, container in self.get_registered_losses(query).items()}

    def print_losses(self, epoch: int, iter: int, precision: int=2, capture: bool=False) -> str | None:
        '''Prints (or, optionally, returns) a string of all the most recent loss values.

        Note: This function uses the last value stored in LossManager.history for each loss.
        As such, this function should generally be called AFTER all of the registered loss
        functions have been run for the batch. Calling it before running some or all of the
        loss funtions could lead to unexpected results.

        By default, this function will print a string of all registered losses and their
        most recent values, tagged with epoch and iter, formatted as:

        "(epoch: {e}, iters: {i}) Loss: {L_1_N}: {L_1_V} {L_2_N}: {L_2_V} ..."
        
        Key:
            e : input epoch
            i : input iter
            L_X_N : Loss X name
            L_X_V : Loss X value
        
        If capture=True, however, the function will return the above formatted string rather
        than printing it. 
        
        If you would instead prefer to get the loss values as a dict, please see: 
            - LossManager.get_loss_values()

        If you are trying to access the LMLossContainer items themselves, please see: 
            - LossManager.get_registered_losses()

        Args:
            epoch : The current epoch value. training_loop_iter+1 in the default train script.
            iter : The current iteration.
            precision : The rounding precision of the loss values as they are added to the
                output string.
            capture : If true, function will return loss values string instead of printing.

        '''
        losses = self.get_loss_values()
        output = f'(epoch: {epoch}, iters: {iter}) Loss:'
        for loss in losses: output += f' {loss}: {losses[loss]}'
        if not capture:
            print(output)
            return None
        else: return output
        
    def register_loss_fn(self, loss_name: str, loss_fn: nn.Module, tags: list[str]=[]) -> None:
        '''Registers a loss function with the loss manager.

        A number of LossManager functions rely on their loss_name being unique, so
        that losses can be easily added, removed, grouped, or shifted around, while
        ensuring that we maintain an immutable lookup value internally. As such, this
        funtion will raise a ValueError if the requested loss_name is already assigned
        to a registered loss, including one created by LossManager.init_from_spec().

        Args:
            loss_name : Name of the loss function (e.g. 'G_GAN', 'D_real').
                This name will be the key that is used to interact with the 
                loss function once it is registered with the LossManager.
            loss_fn : The loss function to register with the LossManager. The loss 
                function can be any type of nn.Module. It can accept 2 to 3 inputs, 
                and must have a forward function that returns a torch.Tensor.
            tag : Tags for the new loss value, used to query the loss objects in 
                various forms.

        Raises:
            ValueError : Requested loss_name is already in use.
        '''
        if not loss_name in self.loss_fns:
            self.loss_fns[loss_name] = LMLossContainer(
                function=loss_fn.to(self.device),
                prev_loss=torch.zeros_like(self.dummy).to(self.device),
                tags=tags)
        else: raise ValueError(f'Loss function name already in use: {loss_name}')
        self.build_history()

    def compute_loss_xy(
            self, 
            loss_name: str, 
            x: torch.Tensor | Any, 
            y: torch.Tensor | Any
        ) -> torch.Tensor:
        '''Runs a loss function and returns the result.

        Also detaches a version of the loss result tensor and moves it to CPU, then
        stores it in self.losses_{"G"|"D"}[loss_name][1]. Additionally, this function
        also computes the mean values of the loss tensor and adds it to the associated 
        self.history entry.

        Args:
            loss_name : The name used when registering the losses, or the registered 
                name of an inbuilt loss function if LossManager was initialized from 
                spec. A list of the losses currently registered with the LossManager
                can be accessed with LossManager.get_registered_losses().
            x : First input for the loss function, usually a torch.Tensor.
            y : Second input for the loss function, usually a torch.Tensor.

        Returns:
            torch.Tensor : Computed loss value.

        Raises:
            KeyError : If the loss name does not match any registered loss.
        '''
        try: loss_entry = self.loss_fns[loss_name]
        except KeyError as e:
            message = ('Invalid loss type. Valid types are: {}')
            raise KeyError(message.format(list(self.loss_fns))) from e
        
        loss_value = loss_entry.function(x, y) # Compute loss value
        self.loss_fns[loss_name].prev_loss = loss_value.clone().detach().cpu() # Update prev loss
        self.history[loss_name].append(loss_value.mean().item())

        return loss_value # Return loss value tensor
    
    def compute_loss_xyz(self) -> torch.Tensor:
        '''Equivelent to compute_loss_xy but for 3 input loss function.
        '''
        raise NotImplementedError('Not yet implemented.')
    