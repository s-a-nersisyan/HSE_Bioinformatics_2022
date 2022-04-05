import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, norm, shapiro, mannwhitneyu, rankdata

'''
df = pd.read_csv("qPCR.tsv", sep="\t", index_col=0)
df.loc["deltaCt"] = df.loc["HPRT1"] - df.loc["CD44"]
print(df)
mean_exp = df.loc["deltaCt"].iloc[:3].mean()
ctrl_exp = df.loc["deltaCt"].iloc[3:].mean()
print(2**(ctrl_exp - mean_exp))

print(ttest_ind(df.loc["deltaCt"].iloc[:3], df.loc["deltaCt"].iloc[3:]))
'''
'''
N = 1000
counter = 0
for i in range(N):
    n = 5
    x = norm(0, 0.3).rvs(n)
    y = norm(1, 0.3).rvs(n)
    p = ttest_ind(x, y)[1]
    if p < 0.05:
        counter += 1

print(counter / N)
'''
'''
df = pd.read_csv("breast_cancer_1000_genes.tsv", sep="\t", index_col=0)
print(df)
sns.kdeplot(df.loc["MT-CO3"])
plt.savefig("MT-CO3.pdf")
plt.close()
sns.kdeplot(df.loc["FOXA1"])
plt.savefig("FOXA1.pdf")
plt.close()

print(shapiro(df.loc["MT-CO3"]))
print(shapiro(df.loc["FOXA1"]))

print(mannwhitneyu([10, 11, 12, 14], [13, 15, 16, 17, 180000000]))
'''

df = pd.DataFrame(np.random.normal(0, 1, size=(20000, 10)))
df["p-value"] = [ttest_ind(df.iloc[i, :5], df.iloc[i, 5:])[1] for i in df.index]
df["padj_Bonf"] = np.minimum(df["p-value"] * len(df), 1)
df["padj_BH"] = np.minimum(df["p-value"] * len(df) / rankdata(df["p-value"]), 1)
df = df.sort_values("p-value")
print(len(df.loc[df["p-value"] < 0.05]))
print(df)
