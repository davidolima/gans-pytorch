import torch
import torch.nn as nn
import torchvision

from abc import ABC

class BaseGenerator(ABC, nn.Module):
    """
    Base class for the Generator, a model that generates 2D images from noise.
    params:
     - img_shape: Input image size.
     - img_channels: Number of input image channels.
     - latent_dim: the size of noise data taken as input.
    """
    def __init__(self, latent_dim, img_size, img_channels):
        super(BaseGenerator, self).__init__()

        self.img_size = img_size
        self.img_channels = img_channels
        self.latent_dim = latent_dim

        self.init_size = 4

        self.linear = nn.Linear(self.latent_dim, self.img_channels*self.init_size*self.init_size)
        self.deconv_blocks = nn.Sequential(
            *self._deconv_block(1, 2),
            *self._deconv_block(2, 4),
            *self._deconv_block(4, self.img_channels),
        )
        self.tanh = nn.Tanh()

    def _deconv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=3, stride=1,
                padding=1, dilation=1,
            ),
            nn.Upsample(
                scale_factor=2,
                mode='nearest',
            ),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.shape[0], -1, self.init_size, self.init_size)
        x = self.deconv_blocks(x)
        x = self.tanh(x)
        return x
