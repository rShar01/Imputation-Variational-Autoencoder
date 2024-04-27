import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("zero_mask.csv")
latent_dim = 15
df = df[df["latentDim"] == latent_dim]
print(df.head())
print(df.columns)
ax = sns.lineplot(df, x="epochs", y="pairModelAccuracy", hue="numMasked")

plt.title(f"Classification Accuracy of Paired VAEs on Fraud Points with LatentDim={latent_dim}")
plt.savefig("test.png")
