import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Sobel(torch.nn.Module):
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
    def __init__(self):
        '''Init for Sobel loss function.

        Defines and registers the Sobel kernels.
        '''
        super().__init__()
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, fake, real):
        '''Forward step for Sobel module.
        
        Converts tensors to grayscale, applies sobel filter, compares result.
        '''
        fake_gray = fake.mean(dim=1, keepdim=True)
        real_gray = real.mean(dim=1, keepdim=True)
        grad_fx = F.conv2d(fake_gray, self.sobel_x, padding=1)
        grad_fy = F.conv2d(fake_gray, self.sobel_y, padding=1)
        grad_rx = F.conv2d(real_gray, self.sobel_x, padding=1)
        grad_ry = F.conv2d(real_gray, self.sobel_y, padding=1)
        grad_fake = torch.sqrt(grad_fx ** 2 + grad_fy ** 2 + 1e-6)
        grad_real = torch.sqrt(grad_rx ** 2 + grad_ry ** 2 + 1e-6)
        return F.l1_loss(grad_fake, grad_real)
    
class Laplacian(torch.nn.Module):
    '''Basically Sobel but with a Laplacian filter rather than a Sobel filter.
    
    This can oftentimes encourage the generator to preserve more fine textural
    details. Good values tend to be around the same as Sobel, maybe a little
    lower. Values that are too high can create additional noise in the output
    and exacerbate any already present checkerboard artifacting.

    References:
    - https://www.nv5geospatialsoftware.com/docs/LaplacianFilters.html
    - https://en.wikipedia.org/wiki/Discrete_Laplace_operator
    '''
    def __init__(self):
        '''Init for Laplacian loss.
        
        Defines and registers a Laplacian kernal.
        '''
        super().__init__()
        kernel = torch.tensor([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=torch.float32)
        kernel = kernel.view(1, 1, 3, 3)
        self.register_buffer('kernel', kernel)

    def forward(self, fake, real):
        '''Forward step for Laplacian module.
        
        Converts tensors to grayscale and applies Laplacian filter, then 
        compares results with L1.
        '''
        fake_lap = F.conv2d(
            fake.mean(dim=1, keepdim=True), 
            self.kernel, padding=1)
        real_lap = F.conv2d(
            real.mean(dim=1, keepdim=True), 
            self.kernel, padding=1)

        return F.l1_loss(fake_lap, real_lap)
    
class VGGPerceptual(nn.Module):
    '''Implements a VGG19-based perceptual loss function.

    Note: Running this loss function for the first time, or registering it with
    a LossManager instance, will install the VGG19 default weights from PyTorch 
    if you do not already have them installed in the Python environment you are 
    running it from.

    Please see `pix2pix_graphical.losses.pix2pix_objective` for more 
    information on VGG-based perceptual loss. Good weight values for this loss
    function vary by task. For the facades dataset, a lambda_vgg of 10.0 and
    a lambda_l1 of 100.0 produces results that are almost indistinguishable
    from the ground truths after 100 epoch + 100 decay epochs. On the 
    cityscapes dataset, similar values can also dramatically increase visual 
    realism, especially early in training.
        
    Datasets (facades/cityscapes):
    - https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/
    '''
    def __init__(self):
        '''Init for VGGPerceptual loss.
        
        Initializes VGG19 with default weights, 
        '''
        super().__init__()
        vgg19_weights = models.VGG19_Weights.DEFAULT
        vgg = models.vgg19(weights=vgg19_weights).features.eval()
        vgg.requires_grad_(False)
        self.blocks = nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:16],])

        self.layer_weights = [1.0, 1.0, 1.0]
        self.L1 = nn.L1Loss()

    def forward(self, fake, real):
        fake, real = fake.clone(), real.clone()
        loss = 0.0
        for i, block in enumerate(self.blocks):
            fake, real = block(fake), block(real)
            loss += self.layer_weights[i] * self.L1(fake, real)
        return loss