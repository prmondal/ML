import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

train_data = pd.read_csv('../dataset/regression-simple/train.csv')
test_data = pd.read_csv('../dataset/regression-simple/test.csv')

train_data = train_data.dropna()
test_data = test_data.dropna()

x_train = np.array(train_data['x'])
y_train = np.array(train_data['y'])

x_train = x_train.reshape(-1,1)
y_train = y_train.reshape(-1,1)

x_test = np.array(test_data['x'])
y_test = np.array(test_data['y'])

x_test = x_test.reshape(-1,1)
y_test = y_test.reshape(-1,1)

model = LinearRegression(normalize=True)
model.fit(x_train, y_train)

predict_val = model.predict(x_test)
error = mean_squared_error(predict_val, y_test)
print(f'MSE: {error}')
print(f'R2 Score: {r2_score(predict_val, y_test)}')

plt.figure(figsize=(10,10))
plt.scatter(x_test,y_test,color='blue',label='original')
plt.plot(x_test,predict_val,color='black',label='pred')
plt.legend()
plt.show()