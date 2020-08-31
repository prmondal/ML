import numpy as np
import tensorflow as tf
from tensorflow.keras import backend, models, layers, losses, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

backend.set_image_data_format('channels_last')
BATCH_SIZE = 64
EPOCHS = 2
classes = ['cat', 'dog']
IMAGE_SIZE = 160
SHUFFLE_BUFFER_SIZE = 1000

IS_TRAINED = True
SAVED_MODEL_NAME = 'final_model_cats_dogs.h5'
SAVED_LOG_DIR = 'logs'

def glimpse_dataset(train_dataset: tf.data.Dataset):
    plt.figure(figsize=(5,5))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    i = 1
    for x, label in train_dataset.take(16):
        plt.subplot(4,4,i)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x)
        plt.xlabel(classes[label])
        i = i + 1

    plt.show()

def glimpse_prediction(test_batch: tf.data.Dataset, prediction):
    plt.figure(figsize=(5,5))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    i = 1
    for x, _ in test_batch.take(16):
        plt.subplot(4,4,i)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x)
        plt.xlabel(classes[np.argmax(prediction[i-1])])
        i = i + 1

    plt.show()

def show_prediction(input, prediction):
    plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(input)
    plt.xlabel(classes[np.argmax(prediction)])
    plt.show()

def get_model(input_shape):
    pretrained_model = MobileNetV2(include_top=False, pooling='avg', input_shape=input_shape)
    pretrained_model.trainable = False

    model = models.Sequential()
    model.add(pretrained_model)

    model.add(layers.Dense(len(classes), activation='softmax'))

    model.summary()
    model.compile(optimizer='adam', metrics='accuracy', loss=losses.SparseCategoricalCrossentropy(from_logits=True))
    return model

def get_callbacks():
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10)
    tensorboard_callback = callbacks.TensorBoard(log_dir=SAVED_LOG_DIR, profile_batch=0)
    return [early_stopping, tensorboard_callback]

def show_training_history(history):
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

def evaluate(model, test_dataset: tf.data.Dataset):
    loss, accuracy = model.evaluate(test_dataset, batch_size=BATCH_SIZE)
    print(f'Loss: {loss}')
    print("Accuracy: {:.2f} %".format(accuracy*100))

def preprocess_data(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    return image, label

def run():
    (train, validation, test), metadata = tfds.load(
        'cats_vs_dogs',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True
    )

    #glimpse_dataset(train)

    train = train.map(preprocess_data)
    validation = validation.map(preprocess_data)
    test = test.map(preprocess_data)
    
    train_batch = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    val_batch = validation.batch(BATCH_SIZE)
    test_batch = test.batch(BATCH_SIZE)

    model = get_model((IMAGE_SIZE, IMAGE_SIZE, 3))

    history = model.fit(train_batch, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=val_batch, callbacks=get_callbacks())
    model.save(SAVED_MODEL_NAME)

    show_training_history(history)
    evaluate(model, test_batch)

    prediction = model.predict(test_batch, batch_size=BATCH_SIZE)
    glimpse_prediction(test_batch.unbatch(), prediction)

def predict_image(image_path):
    image = load_img(image_path)
    image_array = img_to_array(image)
    image_array = tf.image.resize_with_pad(image_array, IMAGE_SIZE, IMAGE_SIZE)
    image_array = image_array / 255.0
    image_array = tf.expand_dims(image_array, axis=0)
    
    model = models.load_model(SAVED_MODEL_NAME)
    prediction = model.predict(image_array)
    show_prediction(image_array[0], prediction)

if __name__ == '__main__':
    if not IS_TRAINED:
        run()
    else:
        predict_image('test_image/dog1.jpg')
        predict_image('test_image/dog2.jpg')
