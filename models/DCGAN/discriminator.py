import torch
import torch.nn as nn
import torchvision

from ..BaseDiscriminator import BaseDiscriminator

class Discriminator(BaseDiscriminator):
    def __init__(self, img_size, img_channels):
        super().__init__(
            img_size=img_size,
            img_channels=img_channels
        )

if __name__ == "__main__":
    D = Discriminator(28, 1).to('cuda')
    print(D)

    pretend_img = torch.rand((8, 1, 28, 28), device='cuda')
    output = D(pretend_img)
    print("output shape:", output.shape)
