import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble  import RandomForestRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

x, y = datasets.fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)

model = RandomForestRegressor(random_state=1)
model.fit(x_train, y_train)

predict_val = model.predict(x_test)
error = mean_squared_error(predict_val, y_test)
print(f'MSE: {error}')
print(f'R2 Score: {r2_score(predict_val, y_test)}')