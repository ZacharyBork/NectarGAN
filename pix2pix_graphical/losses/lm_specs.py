import torch
import torch.nn as nn
from typing import Literal

from pix2pix_graphical.config.config_data import Config
from pix2pix_graphical.losses.losses import SobelLoss, LaplacianLoss
from pix2pix_graphical.losses.lm_data import LMHistory, LMLoss

def pix2pix(
        config: Config,
        dummy: torch.Tensor,
        subspec: Literal['basic', 'extended'] = 'basic'
    ) -> None:
    '''Builds and registers LMLoss objects for pix2pix model.

    mode='basic' (default) will contstruct a pix2pix objective function
    as it was originally defined in Section 4.2 of:

        - Isola et al., *Image-to-Image Translation with Conditional 
            Adversarial Networks*, CVPR 2017.
        - https://arxiv.org/abs/1611.07004

    They found that the best results could generally be derived from a
    combination of L1+cGAN loss. That is defined here as:

        - L1: Pixel-wise loss
            - Generator: 'G_L1'
        - cGAN: Conditional adversarial loss
            - Generator: 'G_GAN'
            - Discriminator: 'D_real', 'D_fake'

    mode='extended' will register all of the losses from mode='basic', but
    it will also register some additional loss functions that can sometimes 
    produce fun and interesting results. These are:

        - SobelLoss:
            Sobel based structural loss, which tries to encourage the 
            generator to better preserve large scale features and patterns.
            - Generator: 'G_SOBEL'
        - LaplacianLoss:
            Laplacian based structural loss, which tries to encourage the 
            generator to preserve smaller textural details and sharp edges.
            - Generator: 'G_LAP'
    '''
    device = config.common.device
    # Creates sanitized dummy tensors for loss structure
    def dummy_tensor(): return dummy.clone().detach().cpu()
    BCE = nn.BCEWithLogitsLoss().to(device)  # G_GAN, D_real, D_fake
    L1 = nn.L1Loss().to(device)              # G_L1 loss
    loss_fns = {
        'G_GAN': LMLoss(
            name='G_GAN', function=BCE, 
            loss_weight=config.loss.lambda_gan,
            last_loss_map=dummy_tensor, 
            history=LMHistory([], []), tags=['G']),
        'G_L1': LMLoss(
            name='G_L1', function=L1, 
            loss_weight=config.loss.lambda_l1,
            last_loss_map=dummy_tensor, 
            history=LMHistory([], []), tags=['G']),
        'D_real': LMLoss(
            name='D_real', function=BCE, loss_weight=None,
            last_loss_map=dummy_tensor, 
            history=LMHistory([], []), tags=['D']),
        'D_fake': LMLoss(
            name='D_fake', function=BCE, loss_weight=None,
            last_loss_map=dummy_tensor, 
            history=LMHistory([], []), tags=['D'])}
    if subspec == 'extended':
        SOBEL = SobelLoss().to(device)
        loss_fns['G_SOBEL'] = LMLoss(
            name='G_SOBEL',function=SOBEL, 
            loss_weight=config.loss.lambda_sobel,
            last_loss_map=dummy_tensor, 
            history=LMHistory([], []), tags=['G'])
        LAP = LaplacianLoss().to(device)
        loss_fns['G_LAP'] = LMLoss(
            name='G_LAP', function=LAP, 
            loss_weight=config.loss.lambda_laplacian,
            last_loss_map=dummy_tensor, 
            history=LMHistory([], []), tags=['G'])
    return loss_fns