import torch
import torch.nn as nn
import torchvision

class Generator(nn.Module):
    """
    Generator: Model takes noise as input and generates a 2D image as output,
    latent_dim: the size of noise data taken as input.
    img_shape: shape of output image.
    """
    def __init__(self, latent_dim, img_size, img_channels):
        super(Generator, self).__init__()

        self.img_size = img_size
        self.img_channels = img_channels
        self.latent_dim = latent_dim

        self.init_size = img_size // 4

        self.linear = nn.Linear(self.latent_dim, self.img_channels*self.init_size*self.init_size)
        self.deconv_blocks = nn.Sequential(
            *self._deconv_block(1, 2),
            *self._deconv_block(2, self.img_channels),
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

if __name__ == "__main__":
    G = Generator(100, 224, 1)
    print(G)

    noise = torch.rand((8, 1, 100))
    output = G(noise)
    print("output shape:", output.shape)
