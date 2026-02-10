from typing import Literal

import torch
import torch.nn.functional as F
from torchvision import models

class LossFunction(torch.nn.Module):
    def __init__(
            self, 
            device: str='cpu',
            dtype: torch.dtype=torch.float32,
            reduction: Literal['none', 'mean', 'sum']='mean',
            metric: Literal['L1', 'L2']='L1'
        ) -> None:
        super(LossFunction, self).__init__()
        self.device = device
        self.dtype = dtype
        self.reduction = reduction

        match metric:
            case 'L1': self.loss_metric = F.l1_loss
            case 'L2': self.loss_metric = F.mse_loss
            case _: raise ValueError(f'Invalid loss metric: {metric}')

    def build_return(self, loss: torch.Tensor) -> torch.Tensor:
        match self.reduction:
            case 'none': return loss
            case 'mean': return loss.mean()
            case 'sum':  return loss.sum()
            case _: raise ValueError(f'Invalid reduction: {self.reduction}')

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        
class KernelLoss(LossFunction):
    def __init__(self, *args, **kwargs) -> None:
        super(KernelLoss, self).__init__(*args, **kwargs)

        self.kernels: dict[str, torch.Tensor] = {} 

    def register_kernel(self, name: str, kernel: list[list[float]]) -> None:
        _kernel = torch.tensor(kernel, dtype=self.dtype).to(self.device)
        self.kernels[name] = _kernel.view(1, 1, 3, 3)

    def eval_kernel(
            self, 
            kernel: str, 
            input_tensor: torch.Tensor,
            padding: int = 1,
            cast: bool = True
        ) -> torch.Tensor:
        result = F.conv2d(input_tensor, self.kernels[kernel], padding=padding)
        if cast: result = result.to(self.device, dtype=self.dtype)
        return result

    def eval_kernels_sqrt(
            self, 
            input_tensor: torch.Tensor, 
            padding: int = 1,
            epsilon: float = 1e-6,
            cast: bool = True
        ) -> torch.Tensor:
        def _eval(key: str) -> torch.Tensor:
            return self.eval_kernel(key, input_tensor, padding, False) ** 2
                
        keys = list(self.kernels.keys())
        total = _eval(keys[0])
        for key in keys[1:]: total += _eval(key)
        result = torch.sqrt(total + epsilon)
        if cast: result = result.to(self.device, dtype=self.dtype)
        return result

    def to_grayscale(self, original: torch.Tensor) -> torch.Tensor:
        gray = original.mean(dim=1, keepdim=True, dtype=self.dtype)
        return gray.to(self.device)

class Sobel(KernelLoss):
    '''Implements a Sobel based structure loss function.

    This loss takes a real and a generated image as tensors, converts them to 
    grayscale, then it applies Sobel filters to each. Then it  uses L1 loss to 
    compute the pixel-wise difference between the two. This tries to encourage 
    the generator to better preserve large scale features and patterns, and can 
    also help reduce blurriness around sharp edges.

    Note: Good loss weight values tend to be in the neighbourhood of ~1-15, 
    sometimes 20. I found that Sobel loss can pretty easily lead to mode
    collapse depending on the task if the lambda is too high. One kind of
    interesting thing I found though is that on the facades dataset from 
    https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/, a combination of 
    relatively high Sobel loss and Laplacian loss, but no traditional L1 loss
    penalty, can lead the generator to create fairly believable images which 
    occasionally exibit some interesting hallucinated details.
    '''
    def __init__(self, *args, **kwargs) -> None:
        '''Init for Sobel loss function.

        Defines and registers the Sobel kernels.
        '''
        super(Sobel, self).__init__(*args, **kwargs)
        self.register_kernel(
            'x', [
                [1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]
            ])
        
        self.register_kernel(
            'y', [
                [1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]
            ])

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        '''Forward step for Sobel module.
        
        Converts tensors to grayscale, applies sobel filter, compares result.
        '''
        grad_fake = self.eval_kernels_sqrt(self.to_grayscale(fake))
        grad_real = self.eval_kernels_sqrt(self.to_grayscale(real))
        
        loss = self.loss_metric(grad_fake, grad_real)
        return self.build_return(loss)

class Laplacian(KernelLoss):
    '''Basically Sobel but with a Laplacian filter rather than a Sobel filter.
    
    This can oftentimes encourage the generator to preserve more fine textural
    details. Good values tend to be around the same as Sobel, maybe a little
    lower. Values that are too high can create additional noise in the output
    and exacerbate any already present checkerboard artifacting.

    References:
    - https://www.nv5geospatialsoftware.com/docs/LaplacianFilters.html
    - https://en.wikipedia.org/wiki/Discrete_Laplace_operator
    '''
    def __init__(self, *args, **kwargs) -> None:
        '''Init for Laplacian loss.
        
        Defines and registers a Laplacian kernal.
        '''
        super(Laplacian, self).__init__(*args, **kwargs)
        self.register_kernel(
            'kernel', [
                [0,  1, 0],
                [1, -4, 1],
                [0,  1, 0]
            ])

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        '''Forward step for Laplacian module.
        
        Converts tensors to grayscale and applies Laplacian filter, then 
        compares results with L1.
        '''
        fake_lap = self.eval_kernel('kernel', self.to_grayscale(fake))
        real_lap = self.eval_kernel('kernel', self.to_grayscale(real))

        loss = self.loss_metric(fake_lap, real_lap)
        return self.build_return(loss)
    
class VGGPerceptual(LossFunction):
    '''Implements a VGG19-based perceptual loss function.

    Note: Running this loss function for the first time, or registering it with
    a LossManager instance, will install the VGG19 default weights from PyTorch 
    if you do not already have them installed in the Python environment you are 
    running it from.

    Please see `nectargan.losses.pix2pix_objective` for more information on 
    VGG-based perceptual loss. Good weight values for this loss function vary 
    by task. For the facades dataset, a lambda_vgg of 10.0 and a lambda_l1 of 
    100.0 produces results that are almost indistinguishable from the ground 
    truths after 100 epoch + 100 decay epochs. On the cityscapes dataset, 
    similar values can also dramatically increase visual realism, especially 
    early in training.
        
    Datasets (facades/cityscapes):
    - https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/
    '''
    def __init__(
            self, 
            layer_weights: list[float] = [1.0, 1.0, 1.0], 
            *args, 
            **kwargs
        ) -> None:
        '''Init for VGGPerceptual loss.
        
        Initializes VGG19 with default weights, 
        '''
        super(VGGPerceptual, self).__init__(*args, **kwargs)
        self.blocks = self._extract_feature_maps()
        assert len(layer_weights) == len(self.blocks)
        self.layer_weights = layer_weights

    def _extract_feature_maps(self) -> torch.nn.ModuleList:
        vgg19_weights = models.VGG19_Weights.DEFAULT
        vgg = models.vgg19(weights=vgg19_weights).features.eval()
        vgg = vgg.to(self.device, dtype=self.dtype)
        vgg.requires_grad_(False)
        return torch.nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:16],])

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        fake, real = fake.clone()[:, :3, :, :], real.clone()[:, :3, :, :]
        loss = 0.0
        for i, block in enumerate(self.blocks):
            fake, real = block(fake), block(real)
            loss += self.layer_weights[i] * self.loss_metric(fake, real)
        return self.build_return(loss)
    
