import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

df = pd.read_csv("BRCA_pam50.tsv", sep="\t", index_col=0)
print(df)
X = df.iloc[:, :-1].to_numpy()
model = PCA()
model.fit(X)

print(model.explained_variance_ratio_[:10])
print(df.columns[np.argsort(np.abs(model.components_[0]))][::-1])
print(df.columns[np.argsort(np.abs(model.components_[1]))][::-1])

X_pca = model.transform(X)
df.iloc[:, :-1] = X_pca
df.columns = [f"PC{i + 1}" for i in range(X.shape[1])] + ["Subtype"]
print(df)

sns.scatterplot(x="PC1", y="PC2", hue="Subtype", data=df)
plt.savefig("PCA_BRCA.pdf")
plt.close()

n = 17
sequences = []
for i in range(n):
    sequence = "A"*i + "T" + "A"*(n - i - 1)
    sequences.append(sequence)

for i in range(n):
    sequence = "C"*i + "G" + "C"*(n - i - 1)
    sequences.append(sequence)

print(sequences)
encoder = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1]
}
for i in range(len(sequences)):
    res = []
    for nucl in sequences[i]:
        res += encoder[nucl]
    sequences[i] = res
print(sequences)
X = np.array(sequences)

model = PCA()
model.fit(X)

print(model.explained_variance_ratio_[:10])

X_pca = model.transform(X)
print(X_pca[:, 0])
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1])
plt.savefig("PCA_nucl.pdf")
