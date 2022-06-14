import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import *
from sklearn.mixture import * 
from sklearn.metrics import *

df = pd.read_csv("BRCA_pam50.tsv", sep="\t", index_col=0)
print(df)
'''
subtypes = df["Subtype"].unique().tolist()
colors = sns.color_palette("tab10")
row_colors = [colors[subtypes.index(s)] for s in df["Subtype"]]

sns.clustermap(df.iloc[:, :-1], method="ward", row_colors=row_colors)
plt.savefig("clustermap.pdf")

'''
'''
X = df.iloc[:, :-1].to_numpy()
y = df["Subtype"].to_numpy()
#model = AgglomerativeClustering(n_clusters=6, linkage="average", affinity="l1")
model = KMeans(n_clusters=6)
y_pred = model.fit_predict(X)
print(adjusted_rand_score(y, y_pred))

s = silhouette_samples(X, y)
print(np.mean(s))
df["Silhouette"] = s
sns.kdeplot(x="Silhouette", hue="Subtype", data=df)
plt.savefig("silhouette_kdes.pdf")
'''
model = GaussianMixture(n_components=2)
model.fit(df[["ESR1"]])
log_p = model.score_samples(df[["ESR1"]])
p = np.exp(log_p)

sns.lineplot(x=df["ESR1"], y=p)
sns.histplot(x="ESR1", data=df, stat="density")
plt.savefig("ESR1.pdf")
