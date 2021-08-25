import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, img_size, num_channels, latent_dim):
        super(Generator, self).__init__()

        self.img_size = img_size
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 1200),
            nn.ReLU(),
            nn.Linear(1200, 1200),
            nn.ReLU(),
            nn.Linear(1200, img_size * img_size * num_channels),
            nn.Sigmoid()
        )

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.uniform_(layer.weight, -0.05, 0.05)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.layers(x)


class LinearMaxOut(nn.Module):
    def __init__(self, input_dim, output_dim, num_pieces):
        super(LinearMaxOut, self).__init__()
        self.output_dim = output_dim
        self.num_pieces = num_pieces
        self.linear = nn.Linear(input_dim, output_dim * num_pieces)
        torch.nn.init.uniform_(self.linear.weight, -0.005, 0.005)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        out = self.linear(x)
        out = out.view(out.size(0), self.num_pieces, self.output_dim)
        out = torch.amax(out, dim=1)
        return out


class Discriminator(nn.Module):
    def __init__(self, img_size, num_channels):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),

            LinearMaxOut(img_size * img_size * num_channels, 240, 5),
            nn.Dropout(0.5),

            LinearMaxOut(240, 240, 5),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(240, 1),
            nn.Sigmoid(),
        )

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.uniform_(layer.weight, -0.005, 0.005)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        out = self.layers(x)
        return out
