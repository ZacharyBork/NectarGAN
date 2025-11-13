import torch
from nectargan.models.unet.model import UnetGenerator
from nectargan.models.patchgan.model import Discriminator
from nectargan.models.resnet.classifier import ResNetClassifier
from nectargan.models.resnet.generator import ResNetGenerator
from nectargan.models.diffusion.denoising_autoencoder import UnetDAE

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

def test_resnet_classifier() -> None:
    '''Tests each ResNetClassifier model variant in its default configuration.
    
    Passes a random tensor to the model and ensures that the output tensor is
    of the expected shape.
    '''
    x = torch.randn(2, 3, 224, 224)
    counts = [50, 101, 152]
    for count in counts:
        net = ResNetClassifier(layer_count=count)
        y = net(x)
        assert y.shape == torch.Size([2, 1000])

def test_resnet_generator() -> None:
    '''Tests the ResNetGenerator model in its default configuration.
    
    Passes a random tensor to the model and ensures that the output tensor is
    the same shape as the input,
    '''
    x = torch.randn(1, 3, 256, 256)
    net = ResNetGenerator()
    y: torch.Tensor = net(x)
    assert y.shape == x.shape

def test_unet_dae() -> None:
    '''Tests the UNetDAE in its default configuration.
    
    Passes a random tensor to the model and ensures that the output tensor is
    the same shape as the input,
    '''
    x = torch.randn(1, 3, 256, 256)
    t = torch.tensor([500])

    dae = UnetDAE(
        device='cpu',
        time_embedding_dimension=128,
        mlp_hidden_dimension=256,
        mlp_output_dimension=128)
    y = dae(x, t)
    assert y.shape == x.shape

