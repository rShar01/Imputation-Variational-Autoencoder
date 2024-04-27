import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from auto_encoder import AutoEncoder

# Creating random noise in a row
def create_random(n):
    return torch.tensor(np.random.normal(loc = 0, scale = 1, size = n)).float()

def mask_index(arr, idxs, val):
    mask_arr = arr.clone()
    mask_arr[:, idxs] = val
    return mask_arr

# Masking df with given value 
def mask_features(df, n, value, random, avoid_last=False):
    df_copy = df.clone()
    
    for index, row in enumerate(df_copy):
        valid_idxs = row.size(0)-1 if avoid_last else row.size(0)

        idx = torch.randperm(valid_idxs)[:n]
        # print("index: " + str(idx))
        if random == True:
            row[idx] = create_random(n = n)
        else: 
            row[idx] = value
        
    return df_copy

# Training process
# Trains on all points in masked_data + unmasked data unless max_N is specified
def train(autoencoder: AutoEncoder, masked_data: pd.DataFrame, unmasaked_data: pd.DataFrame, epochs=100, max_N=None):
    opt = torch.optim.Adam(autoencoder.parameters())
    batch_size = 200
    n, d = unmasaked_data.shape

    avg_los_epoch = []
    for epoch in tqdm(range(epochs)):
        # x = x.to(device) # To format tensors for the GPU
        # Potentially try torch.nn.BCEWithLogitsLoss()
        
        losses = []

        perm_ordering = torch.randperm(n) # potentially control for randomness here
        total = n if max_N is None else max_N

        for i in range(0, total, batch_size):
            curr_indices = perm_ordering[i:i+batch_size]
            X_mask_batch = masked_data[curr_indices]
            X_unmask_batch = unmasaked_data[curr_indices]

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