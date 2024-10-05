import torch
import torch.nn as nn
import torchvision

class Generator(nn.Module):
    """
    Generator: Model takes noise as input and generates a 2D image as output,
    latent_dim: the size of noise data taken as input.
    img_shape: shape of output image.
    """
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()

        img_size, channels = img_shape[1], img_shape[0]
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.l1(x)
        x = x.view(x.shape[0], 128, self.init_size, self.init_size)
        x = self.conv_blocks(x)
        return x


class Discriminator(nn.Module):
    """
    Discriminator: Model takes real images from training data and fake images by Generator and tries seperate them.
    img_shape: Input image shape.
    """
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        img_size, channels = img_shape[1], img_shape[0]

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
            *discriminator_block(channels, 16, normalize=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 512)
        )

        ds_size = img_size // 2**4
        self.output_layer = nn.Sequential(
            nn.Linear(128 * ds_size**2, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.shape[0], -1)
        x = self.output_layer(x)

        return x


class Generator_(nn.Module):
    """
    Generator: Takes noise as input and generates a 2D image as output,
    """
    def __init__(
            self,
            in_channels: int = 3,
            out_size: int = 224,
            out_channels: int = 3,
            latent_dim: int = 100,
    ):
        """
        params:
         - out_size: Output shape for synthetic output.
         - out_channels: Number of channels of synthetic output.
         - latent_dim: Dimension of noise input.
        """
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels

        self._backbone = nn.Sequential()

        # ConvLayers
        size: int = 2
        dim: int = latent_dim
        while dim//2 > out_channels:
            # (B, H, W, C) -> (B, 2*H, 2*W, C/2)
            self._backbone.add_module(
                name=f"DeconvBlock5x5-{dim}->{dim//2}",
                module=self._deconv_block(dim, dim//2, size)
            )
            size *= 2
            dim //= 2

        # Final deconv layer
        self._backbone.add_module(
            name=f"FinalDeconvBlock",
            module=self._deconv_block(dim, out_channels, out_size)
        )

    def _deconv_block(self, in_channels: int, out_channels: int, size: int) -> nn.Module:
        """
        (H,W,C) -> (2*H,2*W,C/2)
        """
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                padding_mode='zeros',
            ),
            nn.LeakyReLU(),
            nn.Upsample(2*size, scale_factor=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B,Z-dim) -> (B,H,W,C)
        """
        #x = x.view(x.shape[0],  self.in_channels, self.latent_dim, -1)
        x = self._backbone(x)
        return x

class Discriminator_(nn.Module):
    """
    Discriminator: Takes real images from training data and fake images by Generator and tries seperate them.
    """
    def __init__(self):
        super(Discriminator_, self).__init__()


    def forward(self, x):
        return x

class WGAN_GP(nn.Module):
    def __init__(
            self,
            img_size: int = 224,
            img_channels: int = 3,
            generator_latent_dim: int = 100,
            n_generator_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.generator: Generator = Generator(
            img_shape=(img_size, img_size, img_channels),
            latent_dim=generator_latent_dim,
        )
        self.discriminator: Discriminator = Discriminator(
            img_shape = (img_size, img_size, img_channels)
        )

    def train_step(self):
        raise Exception("Not implemented yet.")

    def forward_discriminator(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)

    def forward_generator(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

if __name__ == "__main__":
    gan = WGAN_GP(
        img_size=224,
        img_channels=1,
        generator_latent_dim=100,
        n_generator_blocks=3,
    )
    print(gan.generator)
    noise = torch.rand((8, 1, 100))
    generator_output = gan.forward_generator(noise)
    discriminator_output = gan.forward_discriminator(generator_output)
    print(discriminator_output.shape)
