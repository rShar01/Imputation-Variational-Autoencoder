import torch.nn as nn



class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out = out_dim

        # two linear layers good? assuming 1.1*in smaller than out
        hidden_dim = int(1.1*in_dim)
        self.lin1 = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU()
            )
        self.lin2 = nn.Sequential(
                nn.Linear(hidden_dim, out_dim),
                nn.Sigmoid()
            )


    def forward(self, z):
        n, d = z.shape
        
        h1 = self.lin1(z)
        h2 = self.lin2(h1)
        return h2


