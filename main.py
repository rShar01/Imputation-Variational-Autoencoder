import pandas as pd
import torch

from encoder import Encoder




if __name__ == "__main__":
    torch.manual_seed(42)

    df = pd.read_csv("data/credit-card-train.csv")
    y = df["IsFraud"]
    x = df.drop(["id", "IsFraud"], axis=1)

    print(x.head())
    


    enc = Encoder(30, 5)
    some_rows = x.iloc[0:10]
    some_rows = torch.tensor(some_rows.values).float()

    enc.forward(some_rows)
    



