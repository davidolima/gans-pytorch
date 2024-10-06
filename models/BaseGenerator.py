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


    def forward(self, x):
        raise NotImplementedError()
