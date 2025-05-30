import torch.nn as nn
from typing import Literal

from pix2pix_graphical.config.config_data import Config
from pix2pix_graphical.losses.losses import Sobel, Laplacian, VGGPerceptual
from pix2pix_graphical.losses.lm_data import LMLoss, LMWeightSchedule

def pix2pix(
        config: Config,
        subspec: Literal[
            'basic', 
            'basic+vgg', 
            'extended', 
            'extended+vgg'
        ] = 'basic'
    ) -> dict[str, LMLoss]:
    '''Builds LMLoss objects for pix2pix model objective function.

    Note: subspecs that include `+vgg` will install the default VGG19 weights
    from PyTorch if you do not already have them installed.

    subspec='basic' (default) will contstruct a pix2pix objective function as 
    it was originally defined in Section 4.2 of:

        - Isola et al., *Image-to-Image Translation with Conditional 
            Adversarial Networks*, CVPR 2017.
        - https://arxiv.org/abs/1611.07004

    Through their research, they found that the best results could generally 
    be derived from a combination of L1+cGAN loss. That is defined here as:

        - L1: Pixel-wise loss
            - Generator: 'G_L1'
        - cGAN: Conditional adversarial loss
            - Generator: 'G_GAN'
            - Discriminator: 'D_real', 'D_fake'

    subspec='extended' will register all of the losses from subspec='basic', 
    but it will also register some additional loss functions that can sometimes 
    produce fun and interesting results. These are:

        - SobelLoss:
            Sobel based structural loss, which tries to encourage the 
            generator to better preserve large scale features and patterns.
            - Generator: 'G_SOBEL'
        - LaplacianLoss:
            Laplacian based structural loss, which tries to encourage the 
            generator to preserve smaller textural details and sharp edges.
            - Generator: 'G_LAP'

    Both 'basic' and 'extended' have a related subspec called `subspec+vgg`. 
    For each, this will init with the named loss subspec, but it will also 
    register an additional loss, `VGGPerceptual`. VGG loss is really 
    interesting, it passes a real ground truth image and a fake generator 
    result to a pre-trained image classification model, VGG19 in this case, and 
    extracts feature maps for each at various depths. Then it uses a pixel-wise 
    loss (L1 in this implementation) to compare the two. This encourages the 
    generator to make images that are visually similar to the ground truth 
    images, but punishes the generator much less harshly than standard L1 loss 
    for small deviations, allowing the generator some room to create detail 
    without as much blurring or averaging. In some cases, this can allow the 
    generator to create extremely realistic output after a relatively small
    number of epochs. It is computatationally expensive though, and in my 
    extremely informal testing, I noticed around a 10-15% increase in training 
    time with this loss function enabled.

    Resources:
    - https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html
    - https://medium.com/@siddheshb008/vgg-net-architecture-explained-71179310050f
    - https://medium.com/software-dev-explore/neural-style-transfer-vgg19-dab643ec6160
    '''
    device = config.common.device
    BCE = nn.BCEWithLogitsLoss().to(device)  # G_GAN, D_real, D_fake
    L1 = nn.L1Loss().to(device)              # G_L1 loss
    loss_fns = {
        'G_GAN': LMLoss(
            name='G_GAN', function=BCE, 
            loss_weight=config.loss.lambda_gan, tags=['G']),
        'G_L1': LMLoss(
            name='G_L1', function=L1, 
            loss_weight=config.loss.lambda_l1, tags=['G']),
        'D_real': LMLoss(
            name='D_real', function=BCE, tags=['D']),
        'D_fake': LMLoss(
            name='D_fake', function=BCE, tags=['D'])}
    if 'extended' in subspec:
        SOBEL = Sobel().to(device)
        loss_fns['G_SOBEL'] = LMLoss(
            name='G_SOBEL',function=SOBEL, 
            loss_weight=config.loss.lambda_sobel, tags=['G'])
        LAP = Laplacian().to(device)
        loss_fns['G_LAP'] = LMLoss(
            name='G_LAP', function=LAP, 
            loss_weight=config.loss.lambda_laplacian, tags=['G'])
    if '+vgg' in subspec:
        VGG = VGGPerceptual().to(device)
        loss_fns['G_VGG'] = LMLoss(
            name='G_VGG', function=VGG, 
            loss_weight=config.loss.lambda_vgg, tags=['G'])
    return loss_fns