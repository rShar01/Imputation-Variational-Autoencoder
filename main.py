import pandas as pd
import torch
import numpy as np

from tqdm import tqdm
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
def train(autoencoder: AutoEncoder, masked_data: pd.DataFrame, unmasaked_data: pd.DataFrame, epochs=100):
    opt = torch.optim.Adam(autoencoder.parameters())
    batch_size = 10
    n, d = unmasaked_data.shape

    avg_los_epoch = []
    for epoch in tqdm(range(epochs)):
        # x = x.to(device) # To format tensors for the GPU
        # Potentially try torch.nn.BCEWithLogitsLoss()
        
        losses = []

        perm_ordering = torch.randperm(n) # potentially control for randomness here
        for i in range(0, n, batch_size):
            curr_indices = perm_ordering[i:i+batch_size]
            X_mask_batch = masked_data[curr_indices]
            X_unmask_batch = masked_data[curr_indices]

            opt.zero_grad()
            autoencoder.encoder.kl = 0

            X_hat = autoencoder(X_mask_batch)
            loss = ((X_unmask_batch - X_hat)**2).sum() + autoencoder.encoder.kl
            loss = loss / batch_size
            # loss = ((X_unmask_batch - X_hat)**2).sum() 
            losses.append(loss.item())

            loss.backward()
            opt.step()
    
        avg_los_epoch.append(sum(losses)/len(losses))
            
    # print(avg_los_epoch)
    print(f"Staring avg losses: {avg_los_epoch[0]}, {avg_los_epoch[1]}, {avg_los_epoch[2]}")
    print(f"Ending avg loss: {avg_los_epoch[-1]}, {avg_los_epoch[-2]}, {avg_los_epoch[-3]}")
    return autoencoder

if __name__ == "__main__":
    # torch.manual_seed(42)

    df = pd.read_csv("data/credit-card-train.csv")
    df.reset_index(drop=True, inplace=True)

    n,d = df.shape
    df = df.sample(n=n)

    y = df["IsFraud"]
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
    X_subset = X_scaled[:1000]
    X_subset = torch.Tensor(X_subset).float()
    print("input init:\n ", X_subset[0:3])

    train(model, X_subset, X_subset, epochs=600)
    # subset = X_scaled[0:5,:]
    # subset = torch.tensor(subset).float()
    # subset_masked = mask_features(df = subset, n = 5, random = True, value = 0) # Mask random features
    
    # train(model, subset_masked, subset)
    
    preds = model(X_subset[0:3])
    print("output after: \n",preds)
    print(f"MSE: {torch.pow(X_subset[0:3]-preds, 2).sum()/3}")

    
    
# random_mask_df = mask_features(df = subset, n = 5, random = True, value = 9999)
# random_mask_df