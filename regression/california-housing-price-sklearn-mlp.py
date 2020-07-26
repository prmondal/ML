import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

x, y = datasets.fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = MLPRegressor(max_iter=1000, verbose=True)
model.fit(x_train, y_train)

predict_val = model.predict(x_test)
error = mean_squared_error(predict_val, y_test)
print(f'MSE: {error}')
print(f'Test Score: {model.score(x_test, y_test)}')