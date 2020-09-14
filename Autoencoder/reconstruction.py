import numpy as np
import tensorflow as tf
from tensorflow.keras import backend, datasets, models, layers, losses, callbacks
import matplotlib.pyplot as plt

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

backend.set_image_data_format('channels_last')

IMAGE_SIZE = 28
BATCH_SIZE = 64
EPOCHS = 30
SAVED_MODEL_NAME = 'trained_model.h5'

def get_model():
    model = models.Sequential()

    model.add(layers.Conv2D(16, kernel_size=3, strides=2, activation='relu', padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
    model.add(layers.Conv2D(8, kernel_size=3, strides=2, activation='relu', padding='same'))
    model.add(layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'))
    model.add(layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'))
    model.add(layers.Conv2D(1, kernel_size=3, activation='sigmoid', padding='same'))

    model.summary()
    model.compile(optimizer='adam', loss=losses.MeanSquaredError())
    return model

def glimpse_prediction(x_test, prediction, n=10):
    indices = np.random.choice(len(x_test), n)
    plt.subplots(figsize=(20,4))

    for i in range(n):
        plt.subplot(2,n,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.title("original")
        plt.imshow(x_test[indices[i]], cmap='gray')
    
    for i in range(n):
        plt.subplot(2,n,n+i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.title("Reconstructed")
        plt.imshow(prediction[indices[i]], cmap='gray')

    plt.show()

def run():
    (x_train, _), (x_test, _) = datasets.fashion_mnist.load_data()

    x_train = x_train.astype('float32')
    x_train = x_train / 255.0

    x_test= x_test.astype('float32')
    x_test = x_test / 255.0

    x_train = np.reshape(x_train, list(x_train.shape) + [1])
    x_test = np.reshape(x_test, list(x_test.shape) + [1])
    
    model = get_model()
    model.fit(x_train, x_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.3)

    prediction = model.predict(x_test, batch_size=BATCH_SIZE)
    glimpse_prediction(tf.squeeze(x_test), tf.squeeze(prediction))

    model.save(SAVED_MODEL_NAME)

if __name__ == '__main__':
    run()