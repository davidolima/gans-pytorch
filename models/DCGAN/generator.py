from typing import override
import torch
import torch.nn as nn
import torchvision

from models.BaseGenerator import BaseGenerator

class Generator(BaseGenerator):
    def __init__(
            self,
            latent_dim: int,
            img_size: int,
            img_channels: int,
            n_blocks: int = 3,
    ):
        super().__init__(
            latent_dim=latent_dim,
            img_size=img_size,
            img_channels=img_channels
        )

        self.init_size = 4
        self.n_blocks = n_blocks

        self.linear = nn.Linear(self.latent_dim, (2**self.n_blocks)*self.init_size*self.init_size)
        self.resize = torchvision.transforms.Resize((self.img_size,self.img_size))

        self.deconv_layers = nn.Sequential()

        # TODO: Fix? This loop requires that n_blocks > img_channels.
        for i in range(n_blocks, self.img_channels, -1):
            in_channels  = 2**(i)
            out_channels = 2**(i-1)
            in_size = self.init_size*(in_channels//4)
            out_size = self.init_size*(out_channels//4)
            self.deconv_layers.add_module(
                f"Deconv5x5Block_{in_size}->{out_size}",
                self._deconv_block(in_channels, out_channels)
            )

        self.deconv_layers.add_module(
            f"Deconv5x5Block-Final",
            self._deconv_block(2**(self.img_channels), self.img_channels)
        )

    def _deconv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=5, stride=1,
                padding=0, dilation=1,
            ),
            # nn.Upsample(
            #     scale_factor=2,
            #     mode='nearest',
            # ),
            nn.LeakyReLU(),
        )

    @override
    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = x.view(x.shape[0], 2**self.n_blocks, self.init_size, self.init_size)
        x = self.deconv_layers(x)
        x = self.resize(x)
        return x

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath('..'))

    G = Generator(
        latent_dim=100,
        img_size=224,
        img_channels=3,
        n_blocks=3,
    )
    print(G)

    noise = torch.rand((8, 1, 100))
    output = G(noise)
    print("output shape:", output.shape)
