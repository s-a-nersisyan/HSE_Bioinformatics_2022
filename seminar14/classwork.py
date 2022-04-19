import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import *
import networkx as nx
import itertools

df = pd.read_csv("BRCA_pam50.tsv", sep="\t", index_col=0)
print(df)
print(pearsonr(df["UBE2T"], df["BIRC5"]))
print(spearmanr(df["UBE2T"], df["BIRC5"]))
sns.scatterplot(x="UBE2T", y="BIRC5", data=df)
plt.savefig("UBE2T_vs_BIRC5.pdf")
plt.close()
sample = np.random.multivariate_normal([0, 0], [[1, 0.05], [0.05, 1]], size=10000)
print(pearsonr(sample[:, 0], sample[:, 1]))

del df["Subtype"]
df = df.sample(axis=1, frac=1)
r = spearmanr(df, axis=0)[0]
df = pd.DataFrame(r, index=df.columns, columns=df.columns)
print(df)
sns.heatmap(df, cmap=sns.color_palette("coolwarm", as_cmap=True), vmin=-1, vmax=1)
plt.tight_layout()
plt.savefig("heatmap.pdf")
plt.close()
sns.clustermap(df, method="ward", cmap=sns.color_palette("coolwarm", as_cmap=True), vmin=-1, vmax=1)
plt.savefig("clustermap.pdf")
plt.close()

G = nx.Graph()
for g in df.columns:
    G.add_node(g)

for g1, g2 in itertools.combinations(df.columns, 2):
    if np.abs(df.loc[g1, g2]) >= 0.5:
        G.add_edge(g1, g2)

nx.write_graphml(G, "co-expr.graphml")

for nodes in nx.find_cliques(G):
    print(nodes)

n, m = len(G.nodes), len(G.edges)
G = nx.gnm_random_graph(n, m)
for nodes in nx.find_cliques(G):
    print(nodes)
