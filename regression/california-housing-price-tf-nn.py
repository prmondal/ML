import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots

import keras.backend as K
    
def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

BATCH_SIZE = 20
EPOCHS = 200

x, y = datasets.fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = keras.Sequential()
model.add(keras.Input(shape=(x_train.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))

#model.summary()
#keras.utils.plot_model(model, "model.png", show_shapes=True)

model.compile(loss='mse', optimizer='adam', metrics=['mse', soft_acc])

early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, validation_split=0.3, epochs=EPOCHS, callbacks=[early_stop_callback])

_, mse, soft_acc = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print(f'MSE: {mse}')
print("Accuracy: {:.2f} %".format(soft_acc*100))

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot({'Basic': history}, metric = "mse")
plt.ylim([0, 1])
plt.ylabel('MSE')