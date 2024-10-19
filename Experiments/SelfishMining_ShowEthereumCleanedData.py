import numpy as np
from sklearn import linear_model
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

'''
This file displays the specific location of ExperimentData/Blockchair_ethereum_clean.txt (processed data points)
1. Fit a linear function graph
2. Calculate Pearson coefficient
'''

with open("../ExperimentDatas/Blockchair_ethereum_clean.txt", "r") as f:
    m = []
    datas = f.readlines()
    for row in datas:
        s = row.strip().split("\t")
        m.append([int(s[0]), int(s[1]), float(s[2])])
    f.close()

X = np.array(m)


model = linear_model.LinearRegression()
model.fit(X[:,1].reshape(-1, 1), X[:,2])
print("Gradient:", model.coef_)
print("Intercept:", model.intercept_)
print("Pierce coefficient %s"%(str(pearsonr(X[:,2], model.predict(X[:,1].reshape(-1, 1))))))


X_plt = np.arange(0, max(X[:, 1]) + 1).reshape(-1, 1)
y_plt = model.predict(X_plt)

plt.scatter(X[:, 1], X[:, 2], s=10)
plt.plot(X_plt, y_plt, c='red')
plt.grid(ls='--')
plt.show()