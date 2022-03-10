import pandas as pd
import numpy as np

'''
df = pd.DataFrame([[1, "A"], [17, "Q"], [19, "Z"]])
print(df)
print(df.columns)
print(df.index)

df = pd.DataFrame({"A": [1, 3, 4], "B": ["q", "r", "a"]})
print(df)
'''

df = pd.read_csv("TCGA-COAD_cancer_normal.tsv", sep="\t", index_col=0)
gl = pd.read_csv("gene_lengths.tsv", sep="\t", index_col=0).sort_index()
'''
print(df)
print(df[["TCGA-AA-3511-11A", "TCGA-A6-2682-01A"]])
df["smth"] = [i**2 for i in range(len(df))]
print(df)
print(df.loc[["CD44", "NFKB1", "RELA"]])
print(df.loc[["CD44", "NFKB1", "RELA"], ["TCGA-AA-3511-11A", "TCGA-A6-2682-01A"]])
print(df.iloc[5:10])
print(df.iloc[5:10, 1:3])
print(df)
print(df.loc[df["TCGA-A6-2682-01A"] != 0])
print(df.loc[(df["TCGA-A6-2682-01A"] != 0) & (df["TCGA-AA-3514-11A"] != 0)])
print(df.sum(axis=0))
print(df.sum(axis=1))
print(df.loc[df.min(axis=1) > 0])
'''
print(df)
print(gl)

RPK = df.div(gl["Length"], axis=0) * 1000
TPM = RPK.div(RPK.sum(axis=0), axis=1) * 1e+6
print(TPM)
print(TPM.sum(axis=0))

RPM = df.div(df.sum(axis=0), axis=1) * 1e+6
RPKM = RPM.div(gl["Length"], axis=0) * 1000
print(RPKM)
print(RPKM.sum(axis=0))

size_factors = [0.35219656, 0.39439086, 0.73057344, 1.66138079, 1.60002838, 1.48313616, 1.28046971, 0.92434274, 1.59306799, 1.34997698]
RPM = df.div(df.sum(axis=0), axis=1) * 1e+6
DESeq2_RPM = RPM.div(size_factors, axis=1)
DESeq2_RPKM = DESeq2_RPM.div(gl["Length"], axis=0) * 1000
DESeq2_RPKM = np.log2(DESeq2_RPKM + 1)
print(DESeq2_RPKM)

median = DESeq2_RPKM.median(axis=1)
highly_expressed = median.sort_values(ascending=False).iloc[:10000]
DESeq2_RPKM = DESeq2_RPKM.loc[highly_expressed.index]
print(DESeq2_RPKM)
