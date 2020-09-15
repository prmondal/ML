import numpy as np
import tensorflow as tf
from tensorflow.keras import backend, models, layers, losses
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb, gray2rgb
from skimage.transform import resize
from skimage.io import imsave

from tensorflow.keras.applications.vgg16 import VGG16

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

backend.set_image_data_format('channels_last')

np.random.seed(42)

IMAGE_DIR = '../dataset/images/colorization/flowers'
IMAGE_SIZE = 224 # for VGG
BATCH_SIZE = 16
EPOCHS = 350
SAVED_MODEL_PATH = 'trained_model_colorization_350_flower_vgg.h5'
IS_TRAINED = True

# https://github.com/bnsreenu/python_for_microscopists/blob/master/092-autoencoder_colorize_transfer_learning_VGG16_V0.1.py

#### Encoder ####
model_pretrained = VGG16()
encoder = models.Sequential()

for idx, layer in enumerate(model_pretrained.layers):
    if idx <= 18:
        encoder.add(layer)

for layer in encoder.layers:
    layer.trainable = False

#encoder.summary()

#### Decoder ####
decoder = models.Sequential()
decoder.add(layers.Conv2D(256, (3,3), activation='relu', padding='same', input_shape=(7,7,512)))
decoder.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
decoder.add(layers.UpSampling2D((2, 2)))
decoder.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
decoder.add(layers.UpSampling2D((2, 2)))
decoder.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
decoder.add(layers.UpSampling2D((2, 2)))
decoder.add(layers.Conv2D(16, (3,3), activation='relu', padding='same'))
decoder.add(layers.UpSampling2D((2, 2)))
decoder.add(layers.Conv2D(2, (3, 3), activation='tanh', padding='same'))
decoder.add(layers.UpSampling2D((2, 2)))

#decoder.summary()
decoder.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['accuracy'])

def load_dataset():
    data_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
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

def show_image(img):
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img)
    plt.show()

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

    input_img = gray2rgb(img_color)
    input_img = input_img.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

    encoder_output = encoder.predict(input_img)
    decoder_output = model.predict(encoder_output)

    decoder_output = decoder_output * 128 # scale from (-1 to 1) to (-127 to 128) 
    
    result = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
    result[:,:,0] = img_color[0][:,:,0]
    result[:,:,1:] = decoder_output[0]
    result = lab2rgb(result)
    result = tf.image.resize(result, (orig_h, orig_w)) # restore original size

    show_image(result)

def run():
    (x_train, y_train) = load_dataset()
    x_train = np.reshape(x_train, list(x_train.shape) + [1])

    # extract features from pre-trained encoder
    pretrained_features = []

    for img in x_train:
        img = gray2rgb(img)
        img = img.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        
        output_pretrained = encoder.predict(img)
        output_pretrained = output_pretrained.reshape((7,7,512))
        
        pretrained_features.append(output_pretrained)

    pretrained_features = np.array(pretrained_features)

    # use features from pre-trained encoder in decoder as input
    decoder.fit(pretrained_features, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)

    decoder.save(SAVED_MODEL_PATH)

if __name__ == '__main__':
    if IS_TRAINED:
        colorize('test-images/img6.jpg')
    else:
        run()