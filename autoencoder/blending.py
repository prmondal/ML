import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend, models, layers, losses
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb, gray2rgb, rgb2gray
from skimage.transform import resize
from skimage.io import imsave

from tensorflow.keras.applications.vgg16 import VGG16

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

backend.set_image_data_format('channels_last')

np.random.seed(42)

IMAGE_SIZE = 224 # for VGG
PRETRAINED_LAST_LAYER_SIZE = (7,7,512)
EPOCHS = 10000
SAVED_MODEL_PATH_CAT = 'trained_model_cat_vgg16.h5'
SAVED_MODEL_PATH_DOG = 'trained_model_dog_vgg16.h5'
IS_TRAINED = True

TEST_IMAGES_LOCATION = os.path.join(os.getcwd(), 'test-images/blend')

#### Encoder ####
model_pretrained = VGG16()
encoder = models.Sequential()

for idx, layer in enumerate(model_pretrained.layers):
    if idx <= 18:
        encoder.add(layer)

for layer in encoder.layers:
    layer.trainable = False

#encoder.summary()

def get_decoder():
    #### Decoder ####
    # https://github.com/bnsreenu/python_for_microscopists/blob/master/092-autoencoder_colorize_transfer_learning_VGG16_V0.1.py
    decoder = models.Sequential()
    decoder.add(layers.Conv2D(256, (3,3), activation='relu', padding='same', input_shape=PRETRAINED_LAST_LAYER_SIZE))
    decoder.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    decoder.add(layers.UpSampling2D((2, 2)))
    decoder.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    decoder.add(layers.UpSampling2D((2, 2)))
    decoder.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    decoder.add(layers.UpSampling2D((2, 2)))
    decoder.add(layers.Conv2D(16, (3,3), activation='relu', padding='same'))
    decoder.add(layers.UpSampling2D((2, 2)))
    decoder.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))
    decoder.add(layers.UpSampling2D((2, 2)))

    #decoder.summary()
    decoder.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['accuracy'])

    return decoder

decoder1 = get_decoder()
decoder2 = get_decoder()

def load_dataset(image_path):
    data_gen = ImageDataGenerator(
        rescale=1./255
    )

    train_data = data_gen.flow_from_directory(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=100, class_mode=None)

    X = []

    # DO NOT USE ITERATOR ON TRAIN_DATA SINCE IT LOOPS INDEFINITELY
    total_no_batches = len(train_data)
    for idx in range(total_no_batches):
        for img in train_data[idx]:
            X.append(img)
    
    return np.array(X)

def show_image(img):
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img)
    plt.show()

def preprocess_data(image):
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    return image

def train(image_path, encoder, decoder, model_save_path):
    x_train = img_to_array(load_img(image_path))
    x_train = preprocess_data(x_train)
    x_train = tf.expand_dims(x_train, axis=0)

    output_pretrained = encoder.predict(x_train)
    output_pretrained = output_pretrained.reshape(PRETRAINED_LAST_LAYER_SIZE)
    output_pretrained = tf.expand_dims(output_pretrained, axis=0)

    # use features from pre-trained encoder in decoder as input
    decoder.fit(output_pretrained, x_train, epochs=EPOCHS)
    decoder.save(model_save_path)

def test_prediction(image_path, saved_decoder_path):
    decoder = models.load_model(saved_decoder_path)
    img = img_to_array(load_img(image_path))
    img = preprocess_data(img)
    img = tf.expand_dims(img, axis=0)

    encoded_features = encoder.predict(img)
    output = decoder.predict(encoded_features)
    show_image(tf.squeeze(output))

def run():
    CAT_IMAGE_PATH = os.path.join(TEST_IMAGES_LOCATION, 'cat.jpg')
    DOG_IMAGE_PATH = os.path.join(TEST_IMAGES_LOCATION, 'dog.jpg')

    train(CAT_IMAGE_PATH, encoder, decoder1, SAVED_MODEL_PATH_CAT)
    train(DOG_IMAGE_PATH, encoder, decoder2, SAVED_MODEL_PATH_DOG)

if __name__ == '__main__':
    if IS_TRAINED:
        test_image = os.path.join(TEST_IMAGES_LOCATION, 'cat.jpg')
        test_prediction(test_image, SAVED_MODEL_PATH_DOG)

        test_image = os.path.join(TEST_IMAGES_LOCATION, 'dog.jpg')
        test_prediction(test_image, SAVED_MODEL_PATH_CAT)
    else:
        run()