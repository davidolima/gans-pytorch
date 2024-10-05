
import os
from typing import *
from tqdm import tqdm
import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision.utils import save_image

from models.generator import Generator
from models.discriminator import Discriminator

class DCGAN(nn.Module):
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
        super().__init__()

        self.img_size = img_size
        self.img_channels = img_channels
        self.generator_latent_dim = generator_latent_dim
        self.n_generator_blocks = n_generator_blocks

        self.generator: Generator = Generator(
            img_size=img_size,
            img_channels=img_channels,
            latent_dim=generator_latent_dim,
        ).to(device)
        self.gen_optimizer: torch.optim.Optimizer = gen_optimizer(
            params=self.generator.parameters(),
            lr=lr
        )
        print(self.generator)

        self.discriminator: Discriminator = Discriminator(
            img_size=img_size,
            img_channels=img_channels,
        ).to(device)
        self.disc_optimizer: torch.optim.Optimizer = disc_optimizer(
            params=self.discriminator.parameters(),
            lr=lr
        )
        print(self.discriminator)

    def train_step(
            self,
            image_batch: torch.Tensor,
            criterion: Callable,
            device: Literal["cuda","cpu"] = "cuda"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def load_generator_checkpoint(self, cp_path: str):
        state_dict = torch.load(cp_path)
        self.generator.load_state_dict(state_dict['checkpoint'])

    def load_discriminator_checkpoint(self, cp_path: str):
        state_dict = torch.load(cp_path)
        self.discriminator.load_state_dict(state_dict['checkpoint'])

    def train(
            self,
            data_loader: DataLoader,
            output_dir: str,
            n_epochs: int,
            sampling_interval: int = 0,
            device: Literal["cuda", "cpu"] = "cuda",
            save_best_model: bool = True,
    ) -> float:
        """
        Train model on a set amount of epochs.
        Returns best loss reached by generator.
        """
        #writer = SummaryWriter(log_dir=f".logs/dcgan/{strftime('%d%m%Y_%H%M%S')}")
        print(f"[!] Running on {device}.")

        os.makedirs(output_dir, exist_ok=True)

        best_gen_loss = float("inf")

        progress_bar = tqdm(range(n_epochs))
        for epoch in progress_bar:
            mean_gen_loss, mean_disc_loss, batch_size = 0, 0, 0
            for batch, (images, _) in enumerate(tqdm(data_loader)):
                images.to(device)
                fake_images, gen_loss, disc_loss = self.train_step(
                    image_batch=images,
                    criterion=nn.BCELoss(),
                    device=device
                )
                mean_gen_loss  += gen_loss
                mean_disc_loss += disc_loss
                batch_size = len(images)

            mean_gen_loss, mean_disc_loss = mean_gen_loss/batch_size, mean_disc_loss/batch_size
            progress_bar.set_description(f"G Loss: {mean_gen_loss} D Loss: {mean_disc_loss}")

            if save_best_model and mean_gen_loss < best_gen_loss:
                print(f"[!] New best generator loss: {best_gen_loss} -> {mean_gen_loss}.")
                cp_path = os.path.join(output_dir, "weights")
                os.makedirs(cp_path, exist_ok=True)

                self.save_checkpoint(cp_path)

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

        return best_gen_loss

    def save_checkpoint(self, output_path: str):
        output_file = os.path.join(output_path, "best_loss_wgan-gp.pt")
        print(f"[!] Saving weights to `{output_file}`...", end=' ')
        checkpoint = {
            "generator_state_dict": self.generator.state_dict(),
            "gen_optim_state_dict": self.gen_optimizer.state_dict(),

            "discriminator_state_dict": self.discriminator.state_dict(),
            "disc_optim_state_dict": self.gen_optimizer.state_dict(),
        }
        torch.save(checkpoint, output_file)
        print("done.")

    @staticmethod
    def generate_noise(noise_shape: torch.Tensor, device: Literal["cuda", "cpu"]):
        """Returns noise tensor."""
        return torch.autograd.Variable(torch.rand_like(noise_shape, device=device))

    def forward_disc(self, x: torch.Tensor) -> torch.Tensor:
         return self.discriminator(x)

    def forward_gen(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

if __name__ == "__main__":
    gan = DCGAN(
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
