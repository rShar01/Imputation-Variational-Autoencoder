import pandas as pd
import torch
import numpy as np

from sklearn.preprocessing import StandardScaler
from auto_encoder import AutoEncoder 
from training_utils import train


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
    num_points = 200
    is_fraud_idx = y == 1
    not_fraud_idx = y == 0

    X_fraud = X_scaled[is_fraud_idx]
    X_not_fraud = X_scaled[not_fraud_idx]

    X_fraud_tensor = torch.tensor(X_fraud[:num_points]).float()
    X_not_fraud_tensor = torch.tensor(X_not_fraud[:num_points]).float()

    fraud_model = AutoEncoder(input_dim, latent_dim)
    not_fraud_model = AutoEncoder(input_dim, latent_dim)

    print(f"fraud={np.sum(y==1)}\t not={np.sum(y==0)}")

    fraud_model = train(fraud_model, X_fraud_tensor, X_fraud_tensor, epochs=500, max_N=250)
    not_fraud_model = train(not_fraud_model, X_not_fraud_tensor, X_not_fraud_tensor, epochs=500, max_N=250)

    print("Evaluating non fraud reconstruction")
    for i in range(3):
        point = torch.tensor(X_not_fraud[num_points+i]).float()
        # print(point)
        # print(point.shape)

        not_fraud_recon = not_fraud_model(point)
        not_fraud_loss = ((not_fraud_recon - point)**2).sum()

        fraud_recon = fraud_model(point)
        fraud_loss = ((fraud_recon- point)**2).sum()

        print(f"\tpoint {i}:")
        print(f"\t\tfraud loss:     {fraud_loss}")
        print(f"\t\tnot fraud loss: {not_fraud_loss}")

    print("Evaluating fraud reconstruction")
    for i in range(3):
        point = torch.tensor(X_fraud[num_points+i]).float()

        not_fraud_recon = not_fraud_model(point)
        not_fraud_loss = ((not_fraud_recon - point)**2).sum()

        fraud_recon = fraud_model(point)
        fraud_loss = ((fraud_recon- point)**2).sum()

        print(f"\tpoint {i}:")
        print(f"\t\tfraud loss:     {fraud_loss}")
        print(f"\t\tnot fraud loss: {not_fraud_loss}")
