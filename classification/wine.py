import pandas as pd
import numpy as np
from sklearn.ensemble  import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

x, y = datasets.load_wine(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)

model = RandomForestClassifier(random_state=1)
model.fit(x_train, y_train)

predict_val = model.predict(x_test)
acc = accuracy_score(y_test, predict_val)
print("Accuracy: {:.2f} %".format(acc*100))