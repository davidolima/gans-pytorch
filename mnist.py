#!/usr/bin/env python3

import argparse

from torch import cuda
from torch.optim import AdamW
from models.DCGAN.dcgan import DCGAN

from torch.utils.data import DataLoader

import torchvision.transforms as T
from torchvision.datasets import MNIST

DEFAULT_MODEL = "dcgan"
DEFAULT_EPOCHS = 25
DEFAULT_LR = 1e-5
DEFAULT_BATCH_SIZE = 64
DEFAULT_GEN_BLOCKS = 3
DEFAULT_DISC_BLOCKS = 3
DEFAULT_LATENT_DIM = 100
DEFAULT_SAMPLING_INTERVAL = 5
DEFAULT_DONT_SAVE_BEST = False
DEFAULT_DONT_SAVE_LAST = False
DEFAULT_DEVICE = "cuda" if cuda.is_available() else "cpu"

def get_cli_args():
    parser = argparse.ArgumentParser(description="Train GAN on MNIST Dataset.")

    parser.add_argument(
        "-m", "--model",
        type=str, default=DEFAULT_MODEL,
        help=f"Model to be used for training. (Default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "-e", "--epochs",
        type=float, default=DEFAULT_EPOCHS,
        help=f"Number of training iterations over dataset. (Default: {DEFAULT_EPOCHS})"
    )
    parser.add_argument(
        "-lr", "--learning_rate",
        type=float, default=DEFAULT_LR,
        help=f"Learning rate. (Default: {DEFAULT_LR})"
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Number of images to be processed simultaneously. (Default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "-d", "--device",
        type=str, default=DEFAULT_DEVICE,
        help=f"Dimensionality of latent space to be fed into generator. (Default: {DEFAULT_DEVICE})"
    )
    parser.add_argument(
        "--sampling_interval",
        type=int, default=DEFAULT_SAMPLING_INTERVAL,
        help=f"Interval (in epochs) to wait before saving generator samples during training. (Default: {DEFAULT_SAMPLING_INTERVAL})"
    )
    parser.add_argument(
        "--n_gen_blocks",
        type=int, default=DEFAULT_GEN_BLOCKS,
        help=f"Number of deconv blocks to be used in the generator. (Default: {DEFAULT_GEN_BLOCKS})"
    )
    parser.add_argument(
        "--n_disc_blocks",
        type=int, default=DEFAULT_DISC_BLOCKS,
        help=f"Number of conv blocks to be used in the discriminator. (Default: {DEFAULT_DISC_BLOCKS})"
    )
    parser.add_argument(
        "--latent_dim",
        type=int, default=DEFAULT_LATENT_DIM,
        help=f"Dimensionality of latent space to be fed into generator. (Default: {DEFAULT_LATENT_DIM})"
    )
    parser.add_argument(
        "--dont_save_best",
        action="store_true", default=DEFAULT_DONT_SAVE_BEST,
        help=f"If used, best model won't be saved. (Default: {DEFAULT_DONT_SAVE_BEST})"
    )
    parser.add_argument(
        "--dont_save_last",
        action="store_true", default=DEFAULT_DONT_SAVE_LAST,
        help=f"If used, weights won't be saved after the model is done training. (Default: {DEFAULT_DONT_SAVE_LAST})"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = get_cli_args()

    transforms = T.Compose([
        T.PILToTensor()
    ])

    train_set = MNIST(
        root="/datasets/MNIST",
        train=True,
        transform=transforms
    )

    model = DCGAN(
        img_size=28,
        img_channels=1,
        gen_optimizer=AdamW,
        disc_optimizer=AdamW,
        lr=args.learning_rate,
        n_generator_blocks=args.n_gen_blocks,
        n_discriminator_blocks=args.n_disc_blocks,
        generator_latent_dim=args.latent_dim,
    )

    best_loss = model.train(
        data_loader=DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=True,
        ),
        output_dir="./output",
        n_epochs=25,
        sampling_interval=1,
        device=args.device,
        save_best_model=not args.dont_save_best,
        save_on_last_epoch=not args.dont_save_last,
    )

    print("[!] Finished training. Best loss achieved:", best_loss)
