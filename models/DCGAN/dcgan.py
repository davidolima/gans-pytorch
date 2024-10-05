import os
from typing import *
from tqdm import tqdm
import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision.utils import save_image

from ..BaseGAN import BaseGAN
from .generator import Generator
from .discriminator import Discriminator

class DCGAN(BaseGAN):
    def __init__(
            self,
            img_size: int,
            img_channels: int,
            gen_optimizer,
            disc_optimizer,
            lr: float = 1e-5,
            n_generator_blocks: int = 3,
            generator_latent_dim: int = 100,
            device: Literal["cuda", "cpu"] = 'cuda'
    ) -> None:
        super().__init__(
            img_size=img_size,
            img_channels=img_channels,
            generator = Generator(
                generator_latent_dim,
                img_size,img_channels
            ),
            gen_optimizer=gen_optimizer,
            discriminator= Discriminator(
                img_size, img_channels
            ),
            disc_optimizer=disc_optimizer,
            lr=lr,
            n_generator_blocks=n_generator_blocks,
            generator_latent_dim=generator_latent_dim,
            device=device,
        )

if __name__ == "__main__":
    from torch.optim import AdamW

    gan = DCGAN(
        img_size=224,
        img_channels=1,
        gen_optimizer=AdamW,
        disc_optimizer=AdamW,
        generator_latent_dim=100,
        n_generator_blocks=3,
        device="cpu"
    )
    print(gan.generator)
    noise = torch.rand((8, 1, 100))
    generator_output = gan.forward_gen(noise)
    discriminator_output = gan.forward_disc(generator_output)
    print(discriminator_output.shape)
