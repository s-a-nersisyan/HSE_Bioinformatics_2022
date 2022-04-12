import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import *

x = [1, 2, 3, 4, 5, 11, 7, 8, 9, 10, 6, 12, 13, 14]
fc = np.mean(x[:7]) - np.mean(x[7:])
print(fc)

# H0: fc = 0
# H1: fc != 0

N = 10000
H0_fc_distr = []
counter = 0
for _ in range(N):
    np.random.shuffle(x)
    fc_shuf = np.mean(x[:7]) - np.mean(x[7:])
    H0_fc_distr.append(fc_shuf)

    if np.abs(fc_shuf) > np.abs(fc):
        counter += 1

pval = counter / N
print(pval)

sns.histplot(H0_fc_distr, kde=True)
plt.axvline(x=fc, c="red", ls="--")
plt.savefig("H0_fc_distr.pdf")

M = 20000
n = 26
N = 1000
k = 20
pval = hypergeom(M, n, N).sf(k - 1)
print(pval)
pval = fisher_exact([[k, N - k], [n - k, M - n - (N - k)]], alternative="greater")
print(pval)
odds_ratio = (k / (N - k))  /  ((n - k) / (M - n - (N - k)))
print(odds_ratio)

cont_table = [
    [100, 10],  # hospitalized; 100 old, 10 young
    [50, 60],  # non-hospitalized; 50 old, 60 young
]
pval = fisher_exact(cont_table)
print(pval)
