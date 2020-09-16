import numpy as np
import tensorflow as tf
from tensorflow.keras import backend, models, layers, losses
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from skimage.io import imsave

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

backend.set_image_data_format('channels_last')

np.random.seed(42)

IMAGE_DIR = '../dataset/images/colorization/flowers'
IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 500
SAVED_MODEL_PATH = 'trained_model_colorization_500_flower.h5'
IS_TRAINED = False

# https://github.com/bnsreenu/python_for_microscopists/blob/master/090a-autoencoder_colorize_V0.2.py
def get_model():
    model = models.Sequential()

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same', strides=2))
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same', strides=2))
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))

    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(16, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(2, (3, 3), activation='tanh', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))

    model.summary()
    model.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['accuracy'])
    return model

def load_dataset():
    data_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40
    )

    train_data = data_gen.flow_from_directory(IMAGE_DIR, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=100, class_mode=None)

    X = []
    Y = []

    # DO NOT USE ITERATOR ON TRAIN_DATA SINCE IT LOOPS INDEFINITELY
    total_no_batches = len(train_data)
    for idx in range(total_no_batches):
        for img in train_data[idx]:
            img_array = rgb2lab(img)
            L = img_array[:,:,0]
            AB = img_array[:,:,1:] / 128

            X.append(L)
            Y.append(AB)
    
    X = np.array(X)
    Y = np.array(Y)
    return (X, Y)

def colorize(image_path):
    model = models.load_model(SAVED_MODEL_PATH)

    img_color = []
    img = img_to_array(load_img(image_path))
    
    orig_h = img.shape[0]
    orig_w = img.shape[1]

    img = resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    img_color.append(img)
    img_color = np.array(img_color, dtype=float)
    img_color = rgb2lab(1.0/255 * img_color)[:,:,:,0] # extract L channel
    img_color = img_color.reshape(img_color.shape + (1,)) # expand last dim

    output = model.predict(img_color)
    output = output * 128 # from (-1 to 1) to (-127 to 128) 
    
    result = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
    result[:,:,0] = img_color[0][:,:,0]
    result[:,:,1:] = output[0]
    result = lab2rgb(result)
    result = tf.image.resize(result, (orig_h, orig_w)) # restore original size

    imsave("result.jpg", result)

def run():
    (x_train, y_train) = load_dataset()
    x_train = np.reshape(x_train, list(x_train.shape) + [1])

    model = get_model()
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)

    model.save(SAVED_MODEL_PATH)

if __name__ == '__main__':
    if IS_TRAINED:
        colorize('../dataset/colorization-test/sea-3.jpg')
    else:
        run()