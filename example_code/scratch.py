import pandas as pd
from sklearn import linear_model
import numpy as np

dataset = pd.read_csv('data/data.csv')
dataset.head()
model = linear_model.LinearRegression(fit_intercept=False)

X = dataset.loc[:,'input1':'input2']
y = dataset.loc[:,'target']
n = X.shape[0]
model.fit(X[0:(n-1)], y[1:n])
y_hat = model.predict(X[0:(n-1)])
np.std(y_hat-y[1:n])
