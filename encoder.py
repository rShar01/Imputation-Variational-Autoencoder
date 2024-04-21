import torch.nn as nn
import torch.nn.functional as func
import torch.distributions as dists



class Encoder(nn.Module):

    # in_dim = dimension of input
    # out_dim = dimension of latent space
    def __init__(self, in_dim, out_dim):
        super().__init__()

        nn_transform_dim = 100

        self.test_lin = nn.Linear(in_dim, int(0.75*in_dim))

        self.latentize = nn.Sequential(
            nn.Linear(in_dim, int(0.75*in_dim)),
            nn.ReLU(),
            nn.Linear(int(0.75*in_dim), nn_transform_dim),
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

    def forward(self, x):
        print(x.shape)
        n, d = x.shape
        print(f"n: {n}")

        latents = self.latentize(x)

        mu = self.latent_mean(latents)
        sigma = self.latent_var(latents)

        random_point = self.normal.sample(n)
        print(random_point)





