import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import matplotlib.pyplot as plt
import json

import tensorflow as tf
from tensorflow.keras import backend, models, layers, losses, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMAGE_SIZE = 160
imagenet_labels = {}

def load_imagenet_labels():
    global imagenet_labels
    with open('imagenet_label.json', 'r') as json_file:
        imagenet_labels = json.load(json_file)

def show_prediction(input, prediction):
    plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(input)
    plt.xlabel(imagenet_labels[str(np.argmax(prediction))][1])
    plt.show()

def get_model(input_shape):
    model = MobileNetV2(include_top=True, input_shape=input_shape, weights='imagenet')
    #model.summary()
    model.compile(optimizer='adam', metrics='accuracy', loss=losses.SparseCategoricalCrossentropy(from_logits=True))
    return model

def predict_image(image_path):
    image = load_img(image_path)
    image_array = img_to_array(image)
    image_array = tf.image.resize_with_pad(image_array, IMAGE_SIZE, IMAGE_SIZE)
    image_array = image_array / 255.0
    image_array = tf.expand_dims(image_array, axis=0)

    model = get_model((IMAGE_SIZE, IMAGE_SIZE, 3))
    prediction = model.predict(image_array)
    show_prediction(image_array[0], prediction)

if __name__ == '__main__':
    load_imagenet_labels()
    test_images = [('test_images/' + x) for x in os.listdir('test_images')]
    
    for i in test_images:
        predict_image(i)
