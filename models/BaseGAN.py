import os
from typing import *
from tqdm import tqdm
import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision.utils import save_image

from models.BaseGenerator import BaseGenerator
from models.BaseDiscriminator import BaseDiscriminator

from abc import ABC

class BaseGAN(ABC):
    def __init__(
            self,
            img_size: int,
            img_channels: int,
            generator: BaseGenerator,
            gen_optimizer,
            discriminator: BaseDiscriminator,
            disc_optimizer,
            lr: float = 1e-5,
            generator_latent_dim: int = 100,
            device: Literal["cuda", "cpu"] = 'cuda'
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.img_channels = img_channels
        self.generator_latent_dim = generator_latent_dim

        self.generator = generator
        self.generator.to(device)
        self.gen_optimizer: torch.optim.Optimizer = gen_optimizer(
            params=self.generator.parameters(),
            lr=lr
        )
        print(self.generator)

        self.discriminator = discriminator
        self.discriminator.to(device)
        self.disc_optimizer: torch.optim.Optimizer = disc_optimizer(
            params=self.discriminator.parameters(),
            lr=lr
        )
        print(self.discriminator)

    def train_step(
            self,
            image_batch: torch.Tensor,
            criterion: Callable = nn.BCELoss,
            device: Literal["cuda","cpu"] = "cuda"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Method that defines a training step.
        params:
         - image_batch: Tensor of shape (B,C,H,W) of images.
         - criterion: Loss function to be used. (Default: BCE)
         - device: Device where the calculations will be performed; "cuda" or "cpu".
        """
        self.generator.train()
        self.generator.to(device)
        self.discriminator.train()
        self.discriminator.to(device)

        real = torch.ones(image_batch.shape[0], 1, device=device, requires_grad=False)
        fake = torch.zeros(image_batch.shape[0], 1, device=device, requires_grad=False)

        self.gen_optimizer.zero_grad()
        z = self.generate_noise(torch.Tensor(image_batch.shape[0], self.generator_latent_dim), device)
        fake_batch = self.forward_gen(z)
        gen_loss = criterion(self.forward_disc(fake_batch), real)

        gen_loss.backward()
        self.gen_optimizer.step()

        # Train Discriminator
        self.disc_optimizer.zero_grad()
        real_loss = criterion(self.forward_disc(image_batch.float().to(device)), real)
        fake_loss = criterion(self.forward_disc(fake_batch.detach()), fake)
        disc_loss = (real_loss + fake_loss) / 2

        disc_loss.backward()
        self.disc_optimizer.step()

        return fake_batch, gen_loss, disc_loss

    def train(
            self,
            data_loader: DataLoader,
            output_dir: str,
            n_epochs: int,
            sampling_interval: int = 0,
            device: Literal["cuda", "cpu"] = "cuda",
            save_best_model: bool = True,
            save_on_last_epoch: bool = True,
    ) -> float:
        """
        Train model on a set amount of epochs.
        Returns best loss reached by generator.
        params:
         - data_loader: torch.util.data.DataLoader object.
         - output_dir: Path for saving training output.
         - n_epochs: Duration of training in epochs.
         - sampling_interval: Interval (in epochs) for saving the generator's output.
             Images are saved in path `output_dir`/samples/.
             If sampling_interval = 0, samples won't be saved.
        - device: Device where the calculations will be performed; "cuda" or "cpu".
        - save_best_model: Whether to save the best model or not.
             Weights are saved in path `output_dir`/weights/
        - save_on_last_epoch: Whether to save the model after the model finished training or not.
             Weights are saved in path `output_dir`/weights/
        """
        #writer = SummaryWriter(log_dir=f".logs/dcgan/{strftime('%d%m%Y_%H%M%S')}")
        print(f"[!] Running on {device}.")

        os.makedirs(output_dir, exist_ok=True)

        best_gen_loss = float("inf")

        progress_bar = tqdm(range(n_epochs))
        for epoch in progress_bar:
            mean_gen_loss, mean_disc_loss, batch_size = 0, 0, 0
            for images, _ in tqdm(data_loader):
                images.to(device)
                fake_images, gen_loss, disc_loss = self.train_step(
                    image_batch=images,
                    criterion=nn.BCELoss(),
                    device=device
                )
                mean_gen_loss  += gen_loss.item()
                mean_disc_loss += disc_loss.item()
                batch_size = len(images)

            mean_gen_loss, mean_disc_loss = mean_gen_loss/batch_size, mean_disc_loss/batch_size
            progress_bar.set_description(f"G Loss: {mean_gen_loss} D Loss: {mean_disc_loss}")

            if save_best_model and mean_gen_loss < best_gen_loss:
                print(f"[!] New best generator loss: {best_gen_loss} -> {mean_gen_loss}.")
                cp_path = os.path.join(output_dir, "weights")
                os.makedirs(cp_path, exist_ok=True)

                self.save_checkpoint(cp_path, "best_loss_dcgan")

                best_gen_loss = mean_gen_loss

            if sampling_interval != 0 and epoch % sampling_interval == 0:
                sample_path = os.path.join(output_dir, "samples")
                os.makedirs(sample_path, exist_ok=True)
                save_image(
                    fake_images.data[:25],
                    os.path.join(sample_path, f"epoch-{epoch}-{datetime.datetime.today()}.png"),
                    nrow=5,
                    normalize=True,
                )
        if save_on_last_epoch:
            print(f"[!] Training complete, saving last model.")
            cp_path = os.path.join(output_dir, "weights")
            os.makedirs(cp_path, exist_ok=True)

            self.save_checkpoint(cp_path, "last_model")

            best_gen_loss = mean_gen_loss

        return best_gen_loss

    def save_checkpoint(self, output_path: str, fname: str):
        """
        Saves checkpoint to file `fname.pt` in `output_path`.
        """
        fname = fname if fname.endswith('.pt') else fname + '.pt'
        output_file = os.path.join(output_path, fname)
        print(f"[!] Saving weights to `{output_file}`...", end=' ')
        checkpoint = {
            "generator_state_dict": self.generator.state_dict(),
            "gen_optim_state_dict": self.gen_optimizer.state_dict(),

            "discriminator_state_dict": self.discriminator.state_dict(),
            "disc_optim_state_dict": self.gen_optimizer.state_dict(),
        }
        torch.save(checkpoint, output_file)
        print("done.")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Loads GAN checkpoint.
        """
        print(f"[!] Loading checkpoint file...", end=' ')
        checkpoint = torch.load(checkpoint_path)
        self.discriminator.load_state_dict(checkpoint['disc_state_dict'])
        self.disc_optimizer.load_state_dict(checkpoint['disc_optim_state_dict'])

        self.generator.load_state_dict(checkpoint['gen_state_dict'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optim_state_dict'])
        print("done. Weights successfully loaded.")

    @staticmethod
    def generate_noise(noise_shape: torch.Tensor, device: Literal["cuda", "cpu"]):
        """
        Returns noise tensor.
        """
        return torch.autograd.Variable(torch.rand_like(noise_shape, device=device))

    def forward_disc(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through discriminator.
        (B,C,H,W) -> (B, 1)
        """
        return self.discriminator(x)

    def forward_gen(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through generator.
        (B, Z) -> (B,C,H,W)
        """
        return self.generator(x)
