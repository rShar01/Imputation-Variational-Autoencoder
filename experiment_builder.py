from itertools import product


seed_vals = [2,3,13,42,69]
epochs = [30, 50, 100, 200, 300, 500, 1000]
latents = [4, 10, 15]
num_masked = [2, 3, 5]
out = "rand_mask_label_pred_results.csv"


with open("experiment.bash", 'w') as f:
    for seed, ep, lat, num_mask in product(seed_vals, epochs, latents, num_masked):
        f.write(f"!python3 VAE_point_classification.py --seed {seed} --epochs {ep} --latent_dim {lat} --num_masked {num_mask} --csv_loc {out}\n")

