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

        def discriminator_block(in_filters, out_filters, normalize=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if normalize:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(self.img_channels, 4, normalize=False),
            *discriminator_block(4, 8),
            *discriminator_block(8, 16),
        )

        #ds_size = img_size // 2**4
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.output_layer(x)
        return x
