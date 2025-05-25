import torch
import torch.nn as nn
from typing import Literal

from pix2pix_graphical.losses.losses import SobelLoss, LaplacianLoss
from pix2pix_graphical.config.config_data import Config

class LossManager():
    def __init__(
            self, config: Config, 
            init_spec: Literal['core', 'pix2pix'],
            sub_spec: Literal['basic', 'extended']='basic'
        ) -> None:
        self.config = config
        self.device = self.config.common.device
        self.init_spec, self.sub_spec = init_spec, sub_spec
        
        self.make_dummy_shape()
        self.init_from_spec()
        
    def init_from_spec(self) -> None:
        match self.init_spec:
            case 'core': self.losses_G = self.losses_D = {}
            case 'pix2pix': self.register_pix2pix_losses(mode=self.sub_spec)
            case _: raise ValueError(f'Invalid LossManager init_spec: {self.init_spec}')

    def make_dummy_shape(self) -> None:
        '''Creates a dummy tensor of the correct model input shape.
        '''
        channels = self.config.common.input_nc  # Get input channels
        size = self.config.dataloader.crop_size # Get input W, H
        self.dummy = torch.zeros((1, channels, size, size)).to(self.device)
        
    def register_pix2pix_losses(self, mode: Literal['basic', 'extended']='basic') -> None:
        '''Initializes loss values for pix2pix model.'''
        BCE = nn.BCEWithLogitsLoss().to(self.device) # For G_GAN, D_real, D_fake
        L1 = nn.L1Loss().to(self.device)             # Generator L1 loss
        self.losses_G = {
            'GAN': (BCE, torch.zeros_like(self.dummy)),
            'L1': (L1, torch.zeros_like(self.dummy))
        }
        self.losses_D = {
            'real': (BCE, torch.zeros_like(self.dummy)),
            'fake': (BCE, torch.zeros_like(self.dummy))
        }
        if mode == 'extended':
            SOBEL = SobelLoss().to(self.device)
            self.losses_G['SOBEL'] = (SOBEL, torch.zeros_like(self.dummy))
            LAP = LaplacianLoss().to(self.device) 
            self.losses_G['LAP'] = (LAP, torch.zeros_like(self.dummy))

    @property
    def unified_losses(self) -> dict[str, tuple[nn.Module, torch.Tensor]]:
        '''Combines all registed G and D losses for easy lookup.
        
        Returns:
            dict : Dictionary of combined G and D losses.  
        '''
        return self.losses_G | self.losses_D
        
    def register_loss_fn(self, loss_name: str, loss_fn: nn.Module, network: Literal['G', 'D']='G') -> None:
        '''Registers a loss function with the loss manager.

        Args:
            loss_name : Name of the loss function (e.g. 'GAN', 'L1').
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

    def compute_loss_xy(
            self, 
            loss_name: str, 
            x: torch.Tensor, 
            y: torch.Tensor, 
            network: Literal['G', 'D']='G'
        ) -> torch.Tensor:
        '''Runs a loss function and returns the result.
        '''
        match network:
            case 'G': lossmap = self.losses_G
            case 'D': lossmap = self.losses_D
            case _: raise ValueError('Invalid network key. Valid keys: ["D", "G"]')
        
        try: loss_entry = lossmap[loss_name]
        except KeyError as e:
            raise KeyError(
                f'Invalid loss type. Valid types for net{network} are: {list(lossmap)}'
            ) from e
        
        loss_fn = loss_entry[0] # Get loss function
        loss_value = loss_fn(x, y) # Compute loss value
        lossmap[loss_name] = (loss_fn, loss_value) # Update lossmap
        return loss_value
    
    def compute_loss_xyz(self) -> torch.Tensor:
        pass
    



    # def get_loss(
    #         self, loss_fn: str, x: torch.Tensor | None=None, 
    #         y: torch.Tensor | None=None, y_fake: torch.Tensor | None=None) -> torch.Tensor:
    #     '''Runs a loss function and returns the result.
    #     '''
    #     loss_value = torch.zeros_like(self.unified_loss['GAN'])
    #     valid_types = [i for i in self.loss_functions]
    #     try: loss_value = self.loss_functions[loss_fn]
    #     except KeyError as e:
    #         raise KeyError(f'Invalid loss type. Valid type are: {valid_types}') from e
    #     return loss_value

    
