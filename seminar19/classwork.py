import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.metrics import *

df = pd.read_pickle("bc_data.pkl")
an = pd.read_pickle("bc_ann.pkl")

genes = "TRIP13;UBE2C;ZWINT;EPN3;KIF4A;ECHDC2;MTFR1;CX3CR1;SLC7A5;ABAT;CFAP69".split(";")
df = df[genes]

X_train = df.loc[an["Dataset type"] == "Training"].to_numpy()
y_train = an.loc[an["Dataset type"] == "Training", "Class"].to_numpy()

X_test = df.loc[an["Dataset type"] == "Validation"].to_numpy()
y_test = an.loc[an["Dataset type"] == "Validation", "Class"].to_numpy()

for max_depth in range(1, 6):
    model = RandomForestClassifier(max_depth=max_depth, class_weight="balanced", random_state=17)
    model.fit(X_train, y_train)
    
    print(max_depth)
    print(balanced_accuracy_score(y_train, model.predict(X_train)))
    print(balanced_accuracy_score(y_test, model.predict(X_test)))
    print("***")

#plt.figure(figsize=(20, 20))
#plot_tree(model, feature_names=df.columns, fontsize=3)
#plt.savefig("tree.png", dpi=300)
