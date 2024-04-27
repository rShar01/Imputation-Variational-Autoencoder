import pandas as pd
import torch
import numpy as np

import argparse

from sklearn.preprocessing import StandardScaler
from auto_encoder import AutoEncoder
from training_utils import train, initialize_weights


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", action="store_true", default=False)
    parser.add_argument("--display_errors", action="store_true", default=False)

    return parser.parse_args()

if __name__ == "__main__":
    args = parser()
    # torch.manual_seed(42)
    latent_dim = 10

    df = pd.read_csv("data/credit-card-train.csv")
    df_test = pd.read_csv("data/credit-card-test.csv")
    df.reset_index(drop=True, inplace=True)
    exclude_columns = ["id", "Time", "Transaction_Amount"]
    df.drop(exclude_columns, inplace=True, axis=1)

    n,d = df.shape
    df = df.sample(n=n) 

    # train on all features

    all_feat_train_df = df.copy()
    all_feat_train_df["IsFraud"] = -1
    all_feat_train_tens = torch.tensor(all_feat_train_df.to_numpy()).float()
    real_data = torch.tensor(df.iloc[:1000].to_numpy()).float()

    model = AutoEncoder(d, latent_dim)
    # with torch.no_grad():
    #     initialize_weights(model)

    model = train(model, all_feat_train_tens, real_data, epochs=500, max_N=2000)

    eval_num = 100
    correct = 0
    eval_points = df.iloc[1000:1000+eval_num].copy().to_numpy()

    for i in range(eval_num):
        # whichever point achieves
        point = eval_points[i]
        original = point[-1]
        point[-1] = -1
        point = torch.tensor(point).float()
        point = point.reshape(1, d)
        
        x_hat = model(point)

        pred = round(x_hat[0][-1].item())
        if pred == original:
            correct += 1

    print(f"Full model with masking got: {correct/eval_num}")




    # Train by seperating ys
    y = df["IsFraud"].to_numpy()
    X = df.drop(columns=["IsFraud"]).to_numpy()
    n, input_dim = X.shape

    is_fraud_idx = y == 1
    not_fraud_idx = y == 0
    num_fraud_keep = 250
    num_not_fraud_keep = 1000

    X_fraud = X[is_fraud_idx]
    X_not_fraud = X[not_fraud_idx]

    X_fraud_tensor = torch.tensor(X_fraud[:num_fraud_keep]).float()
    X_not_fraud_tensor = torch.tensor(X_not_fraud[:num_not_fraud_keep]).float()

    fraud_model = AutoEncoder(input_dim, latent_dim)
    # initialize_weights(fraud_model)
    not_fraud_model = AutoEncoder(input_dim, latent_dim)
    # initialize_weights(not_fraud_model)

    fraud_model = train(fraud_model, X_fraud_tensor, X_fraud_tensor, epochs=500, max_N=250)
    not_fraud_model = train(not_fraud_model, X_not_fraud_tensor, X_not_fraud_tensor, epochs=500, max_N=250)

    if args.display_errors:
        print("Evaluating non fraud reconstruction")
        for i in range(3):
            point = torch.tensor(X_not_fraud[num_not_fraud_keep+i]).float()
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
            point = torch.tensor(X_fraud[num_fraud_keep+i]).float()

            not_fraud_recon = not_fraud_model(point)
            not_fraud_loss = ((not_fraud_recon - point)**2).sum()

            fraud_recon = fraud_model(point)
            fraud_loss = ((fraud_recon- point)**2).sum()

            print(f"\tpoint {i}:")
            print(f"\t\tfraud loss:     {fraud_loss}")
            print(f"\t\tnot fraud loss: {not_fraud_loss}")


    # fraud cases
    # eval full model 
    X_fraud_with_y = np.c_[X_fraud[0 : num_fraud_keep, :], -1*np.ones(num_fraud_keep)]
    full_model_preds = model(torch.tensor(X_fraud_with_y).float())
    full_model_rounded = torch.round(full_model_preds[:, -1])
    print(f"Full model classification accuracy on fraud points: {torch.sum(full_model_rounded==1).item()}")

    fraud_rest = torch.tensor(X_fraud[0 : num_fraud_keep, :]).float()
    non_fraud_model_pred = not_fraud_model(fraud_rest)
    non_fraud_MSE = torch.sum((fraud_rest - non_fraud_model_pred)**2, 1)

    fraud_model_preds = fraud_model(fraud_rest)
    fraud_MSE = torch.sum((fraud_rest - fraud_model_preds)**2, 1)

    n = fraud_MSE.shape[0]
    correct = 0
    for i in range(n):
        if non_fraud_MSE[i] > fraud_MSE[i]:
            correct += 1
    
    print(f"Paired Discriminating VAE classification accuracy on fraud: {correct/n}")

    # non fraud
    non_fraud_set = torch.tensor(X_not_fraud[num_not_fraud_keep:num_not_fraud_keep+100]).float()

    non_fraud_model_pred = not_fraud_model(non_fraud_set)
    non_fraud_MSE = torch.sum((non_fraud_set - non_fraud_model_pred)**2, 1)

    fraud_model_preds = fraud_model(non_fraud_set)
    fraud_MSE = torch.sum((non_fraud_set - fraud_model_preds)**2, 1)

    n = fraud_MSE.shape[0]
    correct = 0
    for i in range(n):
        if non_fraud_MSE[i] < fraud_MSE[i]:
            correct += 1 

    # print(f"Non fraud test classification: {correct/n}")

    # TODO: do for test points
    test_is_fraud_df = df_test[df_test["IsFraud"] == 1]

