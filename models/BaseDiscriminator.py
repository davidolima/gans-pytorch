import torch
import torch.nn as nn
import torchvision

from abc import ABC

class BaseDiscriminator(nn.Module, ABC):
    """
    Base class for the Discriminator, a model that learns to dicern real images
    from training data and fake images.
    params:
     - img_shape: Input image size.
     - img_channels: Number of input image channels.
    """
    def __init__(self, img_size, img_channels):
        super(BaseDiscriminator, self).__init__()

        self.img_size = img_size
        self.img_channels = img_channels
