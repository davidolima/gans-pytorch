import numpy as np
import torch
import torch.nn as nn
import torchvision

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath('..'))
    from BaseDiscriminator import BaseDiscriminator
else:
    from models.BaseDiscriminator import BaseDiscriminator

class Discriminator(BaseDiscriminator):
    def __init__(
        self,
        img_size: int,
        img_channels: int,
        n_blocks: int = 3,
        starting_dim: int = 8
    ):
        super().__init__(
            img_size=img_size,
            img_channels=img_channels
        )

        self.conv_blocks = nn.Sequential()
        self.conv_blocks.add_module(
            "Conv2d3x3Block_Initial",
            nn.Sequential(
                nn.Conv2d(
                self.img_channels, starting_dim,
                    kernel_size=5, stride=1,
                    padding=0, dilation=1,
                ),
                nn.MaxPool2d(2,2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )

        for _ in range(n_blocks, self.img_channels+1,-1):
            in_channels = starting_dim
            out_channels = starting_dim*2
            in_size = img_size//(in_channels//4)
            out_size = img_size//(out_channels//4)
            self.conv_blocks.add_module(
                f"Conv2d5x5Block_{in_size}->{out_size}",
                self._discriminator_block(in_channels, out_channels)
            )
            starting_dim *= 2

        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear( # (2**self.n_blocks)*self.init_size*self.init_size
                in_features=out_channels*(out_size-(n_blocks))**2,
                out_features=1,
            ),
            nn.Sigmoid()
        )

    def _discriminator_block(self, in_channels, out_channels, dropout=False):
        block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=5, stride=1,
                padding=0, dilation=1,
            ),
            nn.MaxPool2d(2,2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        if dropout:
            block.add_module("Dropout2d", nn.Dropout2d(0.25))
        return block

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.output_layer(x)
        return x

if __name__ == "__main__":
    IMG_SIZE = 28
    IMG_CHANNELS = 1

    D = Discriminator(
        img_size=IMG_SIZE,
        img_channels=IMG_CHANNELS,
        n_blocks = 3
    )
    print(D)

    pretend_img = torch.rand((8, IMG_CHANNELS, IMG_SIZE, IMG_SIZE))
    output = D(pretend_img)
    print("output shape:", output.shape)
