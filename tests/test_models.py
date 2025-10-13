import torch
from nectargan.models.unet.model import UnetGenerator
from nectargan.models.patchgan.model import Discriminator

def test_unet() -> None:
    x = torch.randn((1, 3, 256, 256))
    print(f'Testing UNet model. Input shape: {x.shape}')
    model = UnetGenerator(input_size=512)
    output = model(x)
    result = 'successful' if x.shape == output.shape else 'failed'
    print(f'Test {result}. Output shape: {output.shape}\n')

def test_patchgan() -> None:
    x = torch.randn((1, 3, 64, 64))
    y = torch.randn((1, 3, 64, 64))

    print(f'Testing PatchGAN model. Input shape: {x.shape}')

    model = Discriminator(n_layers=5)
    output = model(x, y)

    result = ('successful' if output.shape == torch.randn((1, 1, 2, 2)).shape 
              else 'failed')
    print(f'Test {result}. Output shape: {output.shape}\n')

if __name__ == "__main__":
    print('Running model validation.\n')
    test_unet()
    test_patchgan()
    print('Model validation complete.')