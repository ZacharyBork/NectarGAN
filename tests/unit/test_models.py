import torch
from nectargan.models.unet.model import UnetGenerator
from nectargan.models.patchgan.model import Discriminator

def test_unet() -> None:
    '''Tests the UNet model in its default configuration.
    
    Passes a random tensor to the model and ensures that the shape of the 
    output matches that of the input.
    '''
    x = torch.randn((1, 3, 256, 256))
    model = UnetGenerator(input_size=512)
    output = model(x)
    assert x.shape == output.shape

def test_patchgan() -> None:
    '''Tests the PatchGAN model in its default configuration.
    
    Passes x/y tensors with shape (1, 3, 64, 64) to the model and asserts that
    the resulting output tensor is the expected shape for the PatchGAN model's
    default layer count.
    '''
    x = torch.randn((1, 3, 64, 64))
    y = torch.randn((1, 3, 64, 64))
    model = Discriminator(n_layers=5)
    output = model(x, y)
    assert output.shape == torch.randn((1, 1, 2, 2)).shape 
