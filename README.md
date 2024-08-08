# Class Based VAE
A ensemble model of VAE models. The dataset is partitioned by each label and a seperate VAE is trained on each partition. This allows the final ensemble model to account for sparse data better than traditional mixture VAEs. This allows the model to perform better on reconstruction tasks (even when the label is missing) and classification tasks.  

See a more detailed write-up [here](https://github.com/rShar01/Imputation-Variational-Autoencoder/blob/main/Class_Based_VAE.pdf)
