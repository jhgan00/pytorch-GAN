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

        self.layers = nn.Sequential(

            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),

            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),

            nn.Linear(512, num_channels * img_size * img_size),
            nn.Tanh()
        )

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                _xavier_init(layer)

    def forward(self, z):
        out = self.layers(z)
        out = out.view(out.size(0), self.num_channels, self.img_size, self.img_size)
        return out


class Discriminator(nn.Module):
    def __init__(self, img_size, num_channels):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_channels * img_size * img_size, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                _xavier_init(layer)

    def forward(self, x):
        out = self.layers(x)
        return out
