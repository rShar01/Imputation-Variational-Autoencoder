import torch.nn as nn
import torch.nn.functional as func




class Encoder(nn.module):

    # in_dim = dimension of input
    # out_dim = dimension of latent space
    def init(self, in_dim, out_dim):
        super().__init__()

        nn_transform_dim = 100

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





