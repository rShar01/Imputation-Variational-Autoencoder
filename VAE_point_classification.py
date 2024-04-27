import pandas as pd
import torch
import numpy as np

import argparse

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from auto_encoder import AutoEncoder
from training_utils import train, mask_features, mask_index


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", action="store_true", default=False)
    parser.add_argument("--display_errors", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--latent_dim", type=int, default=10)
    parser.add_argument("--num_masked", type=int, default=5)
    parser.add_argument("--csv_loc", type=str, default=None, help="Save data as a row in a csv")

    return parser.parse_args()

if __name__ == "__main__":
    args = parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # -- hyperparams -- 
    latent_dim = args.latent_dim
    epochs = args.epochs 

    df = pd.read_csv("data/credit-card-train.csv")
    df.reset_index(drop=True, inplace=True)
    exclude_columns = ["id", "Time", "Transaction_Amount"]
    df.drop(exclude_columns, inplace=True, axis=1)

    n,d = df.shape
    df = df.sample(n=n) 

    num_masked_feat = args.num_masked 
    # mask_idx = [3, 8] # arbitrary, maybe change in experiments?
    mask_idx = torch.randperm(d-1)[:num_masked_feat]

    # train on all features
    all_feat_train_df = df.copy()
    all_feat_train_df["IsFraud"] = -1
    all_feat_train_tens = torch.tensor(all_feat_train_df.to_numpy()).float()
    masked_all_feat_train = mask_features(all_feat_train_tens, num_masked_feat, 0, False, avoid_last=True)
    # masked_all_feat_train = mask_index(all_feat_train_tens, mask_idx, 0)

    real_data = torch.tensor(df.iloc[:1000].to_numpy()).float()

    model = AutoEncoder(d, latent_dim)
    # with torch.no_grad():
    #     initialize_weights(model)

    model = train(model, masked_all_feat_train, real_data, epochs=epochs, max_N=2000)

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
    # masked_fraud_tens = mask_index(X_fraud_tensor, mask_idx, 0)
    masked_fraud_tens = mask_features(X_fraud_tensor, num_masked_feat, 0, False, avoid_last=False)
    X_not_fraud_tensor = torch.tensor(X_not_fraud[:num_not_fraud_keep]).float()
    # masked_not_fraud_tens = mask_index(X_not_fraud_tensor, mask_idx, 0)
    masked_not_fraud_tens = mask_features(X_not_fraud_tensor, num_masked_feat, 0, False, avoid_last=False)

    fraud_model = AutoEncoder(input_dim, latent_dim)
    # initialize_weights(fraud_model)
    not_fraud_model = AutoEncoder(input_dim, latent_dim)
    # initialize_weights(not_fraud_model)

    fraud_model = train(fraud_model, masked_fraud_tens, X_fraud_tensor, epochs=epochs, max_N=250)
    not_fraud_model = train(not_fraud_model, masked_not_fraud_tens, X_not_fraud_tensor, epochs=epochs, max_N=250)

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


    # --  fraud cases --
    # eval full model, fuck it try on every point
    X_fraud_with_y = np.c_[X_fraud, -1*np.ones(X_fraud.shape[0])]
    full_model_preds = model(torch.tensor(X_fraud_with_y).float())
    full_model_rounded = torch.round(full_model_preds[:, -1])
    full_model_right_fraud_preds = torch.sum(full_model_rounded==1).item() 
    print(f"Full model classification num on fraud points: {full_model_right_fraud_preds}")

    # I know there are like 10 points but otherwise we are eval on training points which is just bad
    fraud_rest = torch.tensor(X_fraud[num_fraud_keep:, :]).float()
    non_fraud_model_pred = not_fraud_model(fraud_rest)
    non_fraud_MSE = torch.sum((fraud_rest - non_fraud_model_pred)**2, 1)

    fraud_model_preds = fraud_model(fraud_rest)
    fraud_MSE = torch.sum((fraud_rest - fraud_model_preds)**2, 1)

    n = fraud_MSE.shape[0]
    correct = 0
    for i in range(n):
        if non_fraud_MSE[i] > fraud_MSE[i]:
            correct += 1
    
    pair_discrim_accuracy = correct/n
    print(f"Paired Discriminating VAE classification accuracy on fraud: {pair_discrim_accuracy}")

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
    if args.csv_loc is not None:
        csv_path = Path(args.csv_loc)
        if not csv_path.is_file():
            with open(csv_path, 'w') as f:
                f.write("seed, epochs, latentDim, numMasked, fullModelPredNum, pairModelAccuracy\n")

        with open(csv_path, 'a') as f:
            f.write(f"{args.seed}, {epochs}, {latent_dim}, {num_masked_feat}, {full_model_right_fraud_preds}, {pair_discrim_accuracy}\n")

