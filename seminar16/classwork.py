import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

'''
df = pd.read_csv("BRCA_pam50.tsv", sep="\t", index_col=0)
print(df)

X = df.iloc[:, :-1]
model = TSNE(n_components=2, perplexity=5)
X = model.fit_transform(X)
df["t-SNE 1"] = X[:, 0]
df["t-SNE 2"] = X[:, 1]
sns.scatterplot(x="t-SNE 1", y="t-SNE 2", hue="Subtype", data=df)
plt.savefig("BRCA_tSNE.pdf")
plt.close()
'''
'''
X, y = load_digits(return_X_y=True)
df = pd.DataFrame(X)
df["Digit"] = y
print(df)

model = TSNE(n_components=2, perplexity=50)
X1 = model.fit_transform(X)
df["t-SNE 1"] = X1[:, 0]
df["t-SNE 2"] = X1[:, 1]
sns.scatterplot(x="t-SNE 1", y="t-SNE 2", hue="Digit", data=df)
plt.savefig("digits_tSNE.pdf")
plt.close()

model = PCA(n_components=2)
X2 = model.fit_transform(X)
df["PC 1"] = X2[:, 0]
df["PC 2"] = X2[:, 1]
sns.scatterplot(x="PC 1", y="PC 2", hue="Digit", data=df)
plt.savefig("digits_PCA.pdf")
plt.close()
'''
'''
N = 1000
primes = [2]
for i in range(3, N, 2):
    is_prime = True
    for j in range(3, int(np.sqrt(i)) + 1):
        if i % j == 0:
            is_prime = False
            break
    if is_prime:
        primes.append(i)

X = []
y = []
for i in range(2, N):
    vector = []
    for p in primes:
        vector.append(1 if i % p == 0 else 0)
    X.append(vector)
    y.append(1 if i in primes else 0)

model = TSNE(n_components=2, perplexity=50)
X = model.fit_transform(X)
df = pd.DataFrame()
df["t-SNE 1"] = X[:, 0]
df["t-SNE 2"] = X[:, 1]
df["Type"] = y
sns.scatterplot(x="t-SNE 1", y="t-SNE 2", hue="Type", data=df)
plt.savefig("numbers_tSNE.pdf")
plt.close()
'''

df = pd.read_csv("kegg.tsv", sep="\t")
print(df)

def metric(set1, set2):
    jaccard = len(set1 & set2) / len(set1 | set2)
    return 1 - jaccard

X = []
for i in df.index:
    row = []
    for j in df.index:
        set1 = set(df.loc[i, "Genes"].split(","))
        set2 = set(df.loc[j, "Genes"].split(","))
        row.append(metric(set1, set2))
    X.append(row)

model = TSNE(n_components=2, perplexity=50, metric="precomputed")
X = model.fit_transform(X)
df1 = pd.DataFrame()
df1["t-SNE 1"] = X[:, 0]
df1["t-SNE 2"] = X[:, 1]
sns.scatterplot(x="t-SNE 1", y="t-SNE 2", data=df1)

for i in df.index:
    pathway = df.loc[i, "Pathway"][5:]
    x = X[i, 0]
    y = X[i, 1]
    plt.text(x, y, pathway, size=2)

plt.savefig("KEGG_tSNE.pdf")
plt.close()
