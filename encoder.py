import torch.nn as nn
import torch.nn.functional as func
import torch.distributions as dists

from torch import log as tlog, sum as tsum


class Encoder(nn.Module):

    # in_dim = dimension of input
    # out_dim = dimension of latent space
    # Prob nice preprocessing step: standardize values to N(0,1) so reconstruction easier
    def __init__(self, in_dim, out_dim):
        super().__init__()

        nn_hidden = 15
        self.out = out_dim

        self.latentize = nn.Sequential(
            nn.Linear(in_dim, nn_hidden),
            nn.LeakyReLU(),
        )

        self.latent_mean = nn.Sequential(
                nn.Linear(nn_hidden, out_dim),
                # nn.Tanh()
            )
        
        # Assuming diagonal variances
        self.latent_var = nn.Sequential(
                nn.Linear(nn_hidden, out_dim),
                nn.Sigmoid()  # var should be non-neg
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
        # self.kl += (sigma**2 + mu**2 - tlog(sigma) - 1/2).sum()
        # self.kl += (1 - sigma**2 - mu**2 + tlog(sigma)**2).sum()
        self.kl =  - 0.5 * tsum(1+ tlog(sigma) - mu**2 - sigma)
        return z






