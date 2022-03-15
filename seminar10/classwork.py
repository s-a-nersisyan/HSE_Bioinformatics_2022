from scipy.stats import *
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# Discrete distributions and PMF plot
rv = binom(10, 0.5)
print(rv.mean())
print(rv.var())
print(rv.pmf(0))
print(rv.rvs(size=1000))


df = pd.DataFrame({
    "k": list(range(11)),
    "PMF(k)": [rv.pmf(k) for k in range(11)]
})
print(df)

sns.barplot(x="k", y="PMF(k)", data=df)
plt.savefig("PMF.pdf")

# Continous distributions and PDF plot
rv = norm(2, 1)
print(rv.mean())
print(rv.pdf(2))

df = pd.DataFrame({
    "x": np.linspace(-2, 5, 1000)],
    "PDF(x)": [rv.pdf(x) for x in np.linspace(-2, 5, 1000)]
})
print(df)
sns.lineplot(x="x", y="PDF(x)", data=df)
plt.savefig("PDF.pdf")

# Mathematical expectation for the longest series of ones: Monte-Carlo
n = 100
res = []
for i in range(10000):
    rv = bernoulli(0.5)
    lst = rv.rvs(size=n)
    max_series = 0
    current_series = 0
    for k in lst:
        if k == 1:
            current_series += 1
        else:
            if current_series > max_series:
                max_series = current_series
            current_series = 0
    
    res.append(max_series)

print(res)
print(np.mean(res))
print(np.std(res))

# Calculating integral = pi
rv = uniform(0, 1)
print(np.mean(4*np.sqrt(1 - rv.rvs(size=1000000)**2)))
