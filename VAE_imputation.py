import pandas as pd
import torch
import numpy as np

from sklearn.preprocessing import StandardScaler
from auto_encoder import AutoEncoder 
from training_utils import train, mask_features


if __name__ == "__main__":
    # torch.manual_seed(42)

    df = pd.read_csv("data/credit-card-train.csv")
    df.reset_index(drop=True, inplace=True)

    n,d = df.shape
    df = df.sample(n=n)

    y = df["IsFraud"].to_numpy()
    exclude_columns = ["id", "Time", "Transaction_Amount", "IsFraud"]
    X = df.drop(exclude_columns, axis=1)
    
    # Scale inputs
    # scalar = StandardScaler()
    # X_scaled = scalar.fit_transform(X)
    X_scaled = X.to_numpy()
    
    n, input_dim = X_scaled.shape
    latent_dim = 10
    
    # Instantiate VAE
    model = AutoEncoder(input_dim, latent_dim)
    
    # Try with unmasked data for now
    num_points = 10000
    is_fraud_idx = y == 1
    not_fraud_idx = y == 0

    X_fraud = X_scaled[is_fraud_idx]
    X_not_fraud = X_scaled[not_fraud_idx]

    X_fraud_tensor_training = torch.tensor(X_fraud[:200]).float()
    X_fraud_tensor_testing = torch.tensor(X_fraud[200:]).float()
    X_not_fraud_tensor_training = torch.tensor(X_not_fraud[:num_points]).float()
    X_not_fraud_tensor_testing = torch.tensor(X_not_fraud[num_points:num_points + 50]).float()

    X_fraud_tensor_masked_training = mask_features(X_fraud_tensor_training, 5, 0, random = False)
    X_fraud_tensor_masked_testing = mask_features(X_fraud_tensor_testing, 5, 0, random = False)
    X_not_fraud_tensor_masked_training = mask_features(X_not_fraud_tensor_training, 5, 0, random = False)
    X_not_fraud_tensor_masked_testing = mask_features(X_not_fraud_tensor_testing, 5, 0, random = False)

    fraud_model = AutoEncoder(input_dim, latent_dim)
    not_fraud_model = AutoEncoder(input_dim, latent_dim)
    full_model = AutoEncoder(input_dim, latent_dim)

    print(f"fraud={np.sum(y==1)}\t not={np.sum(y==0)}")

    fraud_model = train(fraud_model, X_fraud_tensor_masked_training, X_fraud_tensor_training, epochs=1000, max_N=None)
    not_fraud_model = train(not_fraud_model, X_not_fraud_tensor_masked_training, X_not_fraud_tensor_training, epochs=1000, max_N=None)
    full_model = train(full_model, torch.vstack((X_fraud_tensor_masked_training,X_not_fraud_tensor_masked_training)), torch.vstack((X_fraud_tensor_training,X_not_fraud_tensor_training)), epochs = 1000, max_N = None)










    print("Evaluating non fraud reconstruction")
    not_fraud_recon = not_fraud_model(X_not_fraud_tensor_masked_testing)
    not_fraud_loss = ((not_fraud_recon - X_not_fraud_tensor_testing)**2).sum()
    not_fraud_loss = not_fraud_loss / X_not_fraud_tensor_testing.shape[0]
    # print(f"\t\tnot fraud loss: {not_fraud_loss.sum(dim = 0)}")

    # print("\tReconstructed masked point:")
    # print(not_fraud_recon)
    # print("\tInput point:")
    # print(point_unmasked)


    print("Evaluating fraud reconstruction")
    fraud_recon = fraud_model(X_fraud_tensor_masked_testing)
    fraud_loss = ((fraud_recon - X_fraud_tensor_testing)**2).sum()
    fraud_loss = fraud_loss / X_fraud_tensor_testing.shape[0]

    print("Total loss on pair of models: " + str((fraud_loss + not_fraud_loss) / 2))

    # fraud_loss = ((fraud_recon- X_fraud_tensor_testing)**2).sum()

    # print(f"\t\tfraud loss:     {fraud_loss}")

    # print("\tReconstructed masked point:")
    # print(fraud_recon)
    # print("\tInput point:")
    # print(point_unmasked)

    print("Evaluating total reconstruction")
    full_recon = full_model(torch.vstack((X_fraud_tensor_masked_testing,X_not_fraud_tensor_masked_testing)))
    full_loss = ((full_recon - torch.vstack((X_fraud_tensor_testing,X_not_fraud_tensor_testing)))**2).sum()
    full_loss = full_loss / torch.vstack((X_fraud_tensor_testing,X_not_fraud_tensor_testing)).shape[0]

    print("Total loss on full model: " + str(full_loss))
    

    print("Average variance of pairwise models: ")
    print(torch.mean(torch.var((fraud_recon - X_fraud_tensor_testing), dim = 0)/2 + torch.var((not_fraud_recon - X_not_fraud_tensor_testing), dim = 0)/2))

    print("Average variance of full model: ")
    print(torch.mean(torch.var((full_recon - torch.vstack((X_fraud_tensor_testing,X_not_fraud_tensor_testing))), dim = 0)))
