import torch.nn as nn

from encoder import Encoder
from decoder import Decoder


# Some code attribution from Alexander Van de Kleut (https://avandekleut.github.io/vae/)

class AutoEncoder(nn.Module):
    def __init__(self, d, latent_dim):
        super().__init__()
        self.encoder = Encoder(d, latent_dim)
        self.decoder = Decoder(latent_dim, d)

    def forward(self, x):
        z = self.encoder.forward(x)
        return self.decoder.forward(z)


