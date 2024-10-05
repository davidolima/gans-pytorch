import torch
import torch.nn as nn
import torchvision

from ..BaseGenerator import BaseGenerator

class Generator(BaseGenerator):
    def __init__(self, latent_dim, img_size, img_channels):
        super().__init__(
            latent_dim=latent_dim,
            img_size=img_size,
            img_channels=img_channels
        )

if __name__ == "__main__":
    G = Generator(100, 224, 1)
    print(G)

    noise = torch.rand((8, 1, 100))
    output = G(noise)
    print("output shape:", output.shape)
