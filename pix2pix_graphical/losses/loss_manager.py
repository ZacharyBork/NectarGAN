import torch
import torch.nn as nn
from typing import Any, Literal

from pix2pix_graphical.losses.losses import SobelLoss, LaplacianLoss
from pix2pix_graphical.config.config_data import Config

class LossManager():
    def __init__(
            self, config: Config, 
            init_spec: Literal['core', 'pix2pix']='core',
            sub_spec: Literal['basic', 'extended']='basic'
        ) -> None:
        self.config = config
        self.device = self.config.common.device
        
        self.make_dummy_shape()
        self.losses_G = self.losses_D = {}
        # self.init_from_spec(init_spec, sub_spec)

    @property
    def unified_losses(self) -> dict[str, tuple[nn.Module, torch.Tensor]]:
        '''Combines all registed G and D losses for easy lookup.
        
        Returns:
            dict : Dictionary of combined G and D losses.  
        '''
        return self.losses_G | self.losses_D
        
    def make_dummy_shape(self) -> None:
        '''Creates a dummy tensor of the correct model input shape.
        '''
        channels = self.config.common.input_nc  # Get input channels
        size = self.config.dataloader.crop_size # Get input W, H
        self.dummy = torch.zeros((1, channels, size, size)).to(self.device)
        
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
            case 'core': self.losses_G = self.losses_D = {}
            case 'pix2pix': self.register_pix2pix_losses(mode=sub_spec)
            case _: raise ValueError(f'Invalid LossManager init_spec: {spec}')
        self.build_history()

    def build_history(self) -> None:
        '''Builds a dict from registered losses to store loss value histories.

        For each loss registered with the LossManager, this function will add an
        entry to the LossManager.history member variable. The key will be the same
        as the lookup key for the corresponding loss function  (e.g. "G_GAN", "D_real"),
        and the value will just be an empty array to append the resulting values of 
        the associate loss function to during training.
        '''
        self.history = {}                # Clear self.history
        for loss in self.unified_losses: # Append loss entries
            self.history[loss] = []      # 1 per registered loss

    def register_pix2pix_losses(self, mode: Literal['basic', 'extended']='basic') -> None:
        '''Initializes loss values for pix2pix model.'''
        # Creates sanitized dummy tensors for loss structure
        dummy_tensor = lambda : self.dummy.clone().detach().cpu()
        BCE = nn.BCEWithLogitsLoss().to(self.device) # For G_GAN, D_real, D_fake
        L1 = nn.L1Loss().to(self.device)             # Generator L1 loss
        self.losses_G = {
            'G_GAN': (BCE, dummy_tensor),
            'G_L1': (L1, dummy_tensor)
        }
        self.losses_D = {
            'D_real': (BCE, dummy_tensor),
            'D_fake': (BCE, dummy_tensor)
        }
        if mode == 'extended':
            SOBEL = SobelLoss().to(self.device)
            self.losses_G['G_SOBEL'] = (SOBEL, dummy_tensor)
            LAP = LaplacianLoss().to(self.device) 
            self.losses_G['G_LAP'] = (LAP, dummy_tensor)
        self.build_history()
    
    def get_registered_losses(
            self, network: Literal['G', 'D'] | None=None
        ) -> dict[str, tuple[nn.Module, torch.Tensor]]:
        '''Returns a dictionary of all losses registered with the LossManager.

        Args:
            network : The network type to get the losses from ["G", "D"], or 
                None (default) to get a dict of registered losses.

        Returns:
            dict : The requested losses. They are returned in the format
                dict[str[nn.Module, torch.Tensor]] where nn.Module is the
                loss function itself, and torch.Tensor is a Tensor storing
                the most recent loss results.
        '''
        match network:
            case 'G': return self.losses_G
            case 'D': return self.losses_D
            case None: return self.unified_losses
            case _: raise ValueError(
                    'Invalid network selection. Valid selections are: Literal["G", "D"]')
            
    def get_loss_values(self, network: Literal['G', 'D'] | None=None, precision: int=2) -> dict[str, float]:
        '''Returns all of the most recent loss values as a dict.
        '''
        def build_output(losses: dict[str, float]):
            output = {} 
            for loss in losses:
                output[loss] = round(self.history[loss][-1], precision)
            return output

        match network:
            case 'G': return build_output(self.losses_G)
            case 'D': return build_output(self.losses_D)
            case None: return build_output(self.unified_losses)
            case _: raise ValueError(
                    'Invalid network selection. Valid selections are: Literal["G", "D"]')
            

        if len(self.unified_losses) == 0:
            raise RuntimeError(f'Unable to print losses: no losses registered.')
        output = {}
        for loss in self.unified_losses:
            output[loss] = round(self.history[loss][-1], precision)
        return output

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
        than printing it. If you would instead prefer to get the loss values as a dict, please
        see: LossManager.get_loss_values().

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
        
    def register_loss_fn(self, loss_name: str, loss_fn: nn.Module, network: Literal['G', 'D']='G') -> None:
        '''Registers a loss function with the loss manager.

        Note: Loss functions are grouped into generator (G) losses and discriminator
        (D) losses. This is done for convenience and organization purposes when 
        interacting with the LossManager. Internally, though, all losses are treated
        the same, and various functions rely on being able to merge the loss types.
        For this reason, no two losses, regardless of associated network may share
        a loss_name. This function will raise a ValueError if the requested name
        already exists for ANY network, not just the one it is being assigned to.

        Args:
            loss_name : Name of the loss function (e.g. 'G_GAN', 'D_real').
                This name will be the key that is used to interact with the 
                loss function once it is registered with the LossManager.
            loss_fn : The loss function to register with the LossManager. The loss 
                function can be any type of nn.Module. It can accept 2 to 3 inputs, 
                and must have a forward function that returns a torch.Tensor.
            network : This value ["G", "D"] deterimines whether the new loss
                function in registered with the generator losses or the
                discriminator losses.

        Raises:
            ValueError : If to_network is invalid. (to_network not in ["G", "D"])
            ValueError : Requested loss_name is already in use.
        '''
        if not loss_name in self.unified_losses:
            new_loss = (
                loss_fn.to(self.device), 
                torch.zeros_like(self.dummy).to(self.device))
            match network:
                case 'G': self.losses_G[loss_name] = new_loss
                case 'D': self.losses_D[loss_name] = new_loss
                case _: raise ValueError('Invalid network key. Valid keys: ["D", "G"]')
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
        lossmap = self.losses_G if loss_name in self.losses_G else self.losses_D
        try: loss_entry = lossmap[loss_name]
        except KeyError as e:
            message = (
                f'Invalid loss type.'
                f'Valid types for net G are: {list(self.losses_G)}'
                f'Valid types for net D are: {list(self.losses_D)}')
            raise KeyError(message) from e
        
        loss_fn = loss_entry[0] # Get loss function
        loss_value = loss_fn(x, y) # Compute loss value
        lossmap[loss_name] = (loss_fn, loss_value.clone().detach().cpu()) # Update lossmap

        self.history[loss_name].append(loss_value.mean().item())
        return loss_value # Return loss value tensor
    
    def compute_loss_xyz(self) -> torch.Tensor:
        '''Equivelent to compute_loss_xy but for 3 input loss function.
        '''
        raise NotImplementedError('Not yet implemented.')
    