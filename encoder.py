import torch.nn as nn
import torch.nn.functional as func
import torch.distributions as dists

from torch import log as tlog


class Encoder(nn.Module):

    # in_dim = dimension of input
    # out_dim = dimension of latent space
    # Prob nice preprocessing step: standardize values to N(0,1) so reconstruction easier
    def __init__(self, in_dim, out_dim):
        super().__init__()

        nn_transform_dim = 100
        nn_hidden = 75
        
        self.out = out_dim

        # 0.75 arbitrary right now but prob good idea for smaller layer
        self.latentize = nn.Sequential(
            nn.Linear(in_dim, nn_hidden),
            nn.ReLU(),
            nn.Linear(nn_hidden, nn_transform_dim),
            nn.ReLU()
        )

        self.latent_mean = nn.Sequential(
                nn.Linear(nn_transform_dim, out_dim),
                nn.Tanh()
            )
        
        # Assuming diagonal variances
        self.latent_var = nn.Sequential(
                nn.Linear(nn_transform_dim, out_dim),
                nn.ReLU() 
            )

        self.normal = dists.Normal(0,1)
        self.kl = 0

    def forward(self, x):
        # print(x.shape)
        # n, d = x.shape
        # print(f"n: {n}")
        
        n = x.shape[0]

        latents = self.latentize(x)

        mu = self.latent_mean(latents)
        sigma = self.latent_var(latents)

        random_point = self.normal.sample(sample_shape=(n,self.out))
        z = mu + sigma*random_point
        self.kl = (sigma**2 + mu**2 - tlog(sigma) - 1/2).sum()
        return z






