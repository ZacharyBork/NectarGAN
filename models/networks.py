import torch
import torch.nn as nn
import torch.nn.functional as F
    
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=4, 
                stride=stride, 
                padding=1, 
                bias=False, 
                padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
class SobelLoss(torch.nn.Module):
    def __init__(self):
        super(SobelLoss, self).__init__()
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, fake, real):
        fake_gray = fake.mean(dim=1, keepdim=True)
        real_gray = real.mean(dim=1, keepdim=True)
        grad_fx = F.conv2d(fake_gray, self.sobel_x, padding=1)
        grad_fy = F.conv2d(fake_gray, self.sobel_y, padding=1)
        grad_rx = F.conv2d(real_gray, self.sobel_x, padding=1)
        grad_ry = F.conv2d(real_gray, self.sobel_y, padding=1)
        grad_fake = torch.sqrt(grad_fx ** 2 + grad_fy ** 2 + 1e-6)
        grad_real = torch.sqrt(grad_rx ** 2 + grad_ry ** 2 + 1e-6)
        return F.l1_loss(grad_fake, grad_real)