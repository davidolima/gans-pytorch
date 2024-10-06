#!/usr/bin/env python3

from torch.optim import AdamW
from models.DCGAN.dcgan import DCGAN

from torch.utils.data import DataLoader

import torchvision.transforms as T
from torchvision.datasets import MNIST

if __name__ == "__main__":
    model = DCGAN(
        img_size=28,
        img_channels=1,
        gen_optimizer=AdamW,
        disc_optimizer=AdamW,
        lr=1e-5,
        n_generator_blocks=3,
        n_discriminator_blocks=3,
        generator_latent_dim=100,
    )

    transforms = T.Compose([
        T.PILToTensor()
    ])

    train_set = MNIST(
        root="/datasets/MNIST",
        train=True,
        transform=transforms
    )

    best_loss = model.train(
        data_loader=DataLoader(
            dataset=train_set,
            batch_size=128,
            shuffle=True,
        ),
        output_dir="./output",
        n_epochs=25,
        sampling_interval=1,
        device="cuda",
        save_best_model=True,
        save_on_last_epoch=True,
    )

    print("[!] Finished training. Best loss achieved:", best_loss)
