import torch
from torch import nn


def _xavier_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, img_size, num_channels, latent_dim):
        super(Generator, self).__init__()

        self.img_size = img_size
        self.num_channels = num_channels
        self.latent_dim = latent_dim


        self.model = nn.Sequential(

            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, num_channels * img_size * img_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), self.num_channels, self.img_size, self.img_size)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_size, num_channels):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(num_channels * img_size * img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
