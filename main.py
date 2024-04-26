import pandas as pd
import torch
import numpy as np

from auto_encoder import AutoEncoder 
from sklearn.preprocessing import StandardScaler

# Creating random noise in a row
def create_random(n):
    return torch.tensor(np.random.normal(loc = 0, scale = 1, size = n)).float()

# Masking df with given value 
def mask_features(df, n, value, random):
    df_copy = df.clone()
    
    for index, row in enumerate(df_copy):
        
        idx = torch.randperm(row.size(0))[:n]
        print("index: " + str(idx))
        if random == True:
            row[idx] = create_random(n = n)
        else: 
            row[idx] = value
        
    return df_copy

# Training process
def train(autoencoder, masked_data, unmasaked_data, epochs=1):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        print("epoch: " + str(epoch))
        for index, x in enumerate(masked_data):
            print("row number: " + str(index))
            # x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            print(x_hat)
            x_clean = unmasaked_data[index]
            
            
            # Trying loss function
            # criterion = torch.nn.BCEWithLogitsLoss()
            # loss = criterion(x_hat, x_clean)
            
            loss = ((x_clean - x_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()
            
    return autoencoder

if __name__ == "__main__":
    # torch.manual_seed(42)

    df = pd.read_csv("data/credit-card-train.csv")
    df.reset_index(drop=True)
    y = df["IsFraud"]
    exclude_columns = ["id", "Time", "Transaction_Amount", "IsFraud"]
    X = df.drop(exclude_columns, axis=1)
    
    scalar = StandardScaler()
    X_scaled = scalar.fit_transform(X)
    
    n, input_dim = X_scaled.shape
    latent_dim = 10
    
    model = AutoEncoder(input_dim, latent_dim)
    
    subset = X_scaled[0:5,:]
    subset = torch.tensor(subset).float()
    subset_masked = mask_features(df = subset, n = 5, random = False, value = 0)
    
    train(model, subset_masked, subset)
    
    print("input masked: ", subset_masked)
    print("input unmasked: ", subset)
    print("output: ", model(subset_masked))

    
    
# random_mask_df = mask_features(df = subset, n = 5, random = True, value = 9999)
# random_mask_df