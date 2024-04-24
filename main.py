import pandas as pd
import torch

from auto_encoder import AutoEncoder 
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    torch.manual_seed(42)

    df = pd.read_csv("data/credit-card-train.csv")
    df.reset_index(drop=True)
    y = df["IsFraud"]
    x = df.drop(["id", "IsFraud", "Time"], axis=1)
    
    scalar = StandardScaler()
    x_scaled = scalar.fit_transform(x)
    print(x.columns)
    
    n, d = x.shape
    
    model = AutoEncoder(d, 10)
    some_rows = x.iloc[0:3]
    some_rows = torch.tensor(some_rows.values).float()

    print(model.forward(some_rows))
    



