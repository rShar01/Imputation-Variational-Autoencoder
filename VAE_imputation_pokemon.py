import pandas as pd
import torch
import numpy as np

from sklearn.preprocessing import StandardScaler
from auto_encoder import AutoEncoder 
from training_utils import train, mask_features

if __name__ == "__main__":
    # torch.manual_seed(42)

    # Load the dataset
    df = pd.read_csv("data/pokemon-train.csv")
    df.reset_index(drop=True, inplace=True)

    n, d = df.shape
    df = df.sample(n=n)
    df.fillna(0, inplace=True)


    # Using 'is_legendary' column as the label
    y = df["is_legendary"].to_numpy()
    exclude_columns = ["is_legendary"]  # Update column names as per your dataset
    X = df.drop(exclude_columns, axis=1)

    # Scale inputs
    scalar = StandardScaler()
    X_scaled = scalar.fit_transform(X)
    # X_scaled = X.to_numpy()
    
    n, input_dim = X_scaled.shape
    latent_dim = 10
    
    # Instantiate AutoEncoder
    model = AutoEncoder(input_dim, latent_dim)
    
    # Indexing for legendary and non-legendary
    is_legendary_idx = y == 1
    not_legendary_idx = y == 0

    X_legendary = X_scaled[is_legendary_idx]
    X_not_legendary = X_scaled[not_legendary_idx]

    X_legendary_tensor_training = torch.tensor(X_legendary[:200]).float()
    X_legendary_tensor_testing = torch.tensor(X_legendary[200:]).float()
    X_not_legendary_tensor_training = torch.tensor(X_not_legendary[:500]).float()
    X_not_legendary_tensor_testing = torch.tensor(X_not_legendary[500:550]).float()

    X_legendary_tensor_masked_training = mask_features(X_legendary_tensor_training, 5, 0, random=False)
    X_legendary_tensor_masked_testing = mask_features(X_legendary_tensor_testing, 5, 0, random=False)
    X_not_legendary_tensor_masked_training = mask_features(X_not_legendary_tensor_training, 5, 0, random=False)
    X_not_legendary_tensor_masked_testing = mask_features(X_not_legendary_tensor_testing, 5, 0, random=False)

    legendary_model = AutoEncoder(input_dim, latent_dim)
    not_legendary_model = AutoEncoder(input_dim, latent_dim)
    full_model = AutoEncoder(input_dim, latent_dim)

    print(f"legendary={np.sum(y==1)}\t not_legendary={np.sum(y==0)}")

    legendary_model = train(legendary_model, X_legendary_tensor_masked_training, X_legendary_tensor_training, epochs=1000, max_N=None)
    not_legendary_model = train(not_legendary_model, X_not_legendary_tensor_masked_training, X_not_legendary_tensor_training, epochs=1000, max_N=None)
    full_model = train(full_model, torch.vstack((X_legendary_tensor_masked_training,X_not_legendary_tensor_masked_training)), torch.vstack((X_legendary_tensor_training,X_not_legendary_tensor_training)), epochs=1000, max_N=None)

    print("Evaluating non-legendary reconstruction")
    not_legendary_recon = not_legendary_model(X_not_legendary_tensor_masked_testing)
    not_legendary_loss = ((not_legendary_recon - X_not_legendary_tensor_testing)**2).sum()
    not_legendary_loss = not_legendary_loss / X_not_legendary_tensor_testing.shape[0]

    print("Evaluating legendary reconstruction")
    legendary_recon = legendary_model(X_legendary_tensor_masked_testing)
    legendary_loss = ((legendary_recon - X_legendary_tensor_testing)**2).sum()
    legendary_loss = legendary_loss / X_legendary_tensor_testing.shape[0]

    print("Total loss on pair of models: " + str((legendary_loss + not_legendary_loss) / 2))

    print("Evaluating total reconstruction")
    full_recon = full_model(torch.vstack((X_legendary_tensor_masked_testing, X_not_legendary_tensor_masked_testing)))
    full_loss = ((full_recon - torch.vstack((X_legendary_tensor_testing, X_not_legendary_tensor_testing)))**2).sum()
    full_loss = full_loss / torch.vstack((X_legendary_tensor_testing, X_not_legendary_tensor_testing)).shape[0]

    print("Total loss on full model: " + str(full_loss))

    print("Average variance of pairwise models: ")
    print(torch.mean(torch.var((legendary_recon - X_legendary_tensor_testing), dim=0) / 2 + torch.var((not_legendary_recon - X_not_legendary_tensor_testing), dim=0) / 2))

    print("Average variance of full model: ")
    print(torch.mean(torch.var((full_recon - torch.vstack((X_legendary_tensor_testing, X_not_legendary_tensor_testing))), dim=0)))
