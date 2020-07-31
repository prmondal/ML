import numpy as np
import tensorflow as tf
from tensorflow.keras import backend, datasets, models, layers, losses, callbacks
import matplotlib.pyplot as plt

backend.set_image_data_format('channels_last')
BATCH_SIZE = 64
EPOCHS = 100
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
IS_TRAINED = True

def glimpse_dataset(x, labels):
    plt.figure(figsize=(5,5))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i])
        plt.xlabel(classes[labels[i][0]])

    plt.show()

def glimpse_prediction(x, prediction):
    plt.figure(figsize=(5,5))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i])
        plt.xlabel(classes[np.argmax(prediction[i])])

    plt.show()

def show_prediction(input, prediction):
    plt.figure(figsize=(1,1))
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(input)
    plt.xlabel(classes[np.argmax(prediction)])
    plt.show()

def get_model(input_shape):
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.4))

    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(len(classes), activation='softmax'))

    model.summary()
    model.compile(optimizer='adam', metrics='accuracy', loss=losses.SparseCategoricalCrossentropy(from_logits=True))
    return model

def get_callbacks():
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10)
    tensorboard_callback = callbacks.TensorBoard(log_dir='logs', profile_batch=0)
    return [early_stopping, tensorboard_callback]

def show_training_history(history):
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

def evaluate(model, x, labels):
    loss, accuracy = model.evaluate(x, labels, batch_size=BATCH_SIZE)
    print(f'Loss: {loss}')
    print("Accuracy: {:.2f} %".format(accuracy*100))

def run():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    x_train = x_train.astype('float32')
    x_test= x_test.astype('float32')
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = get_model(x_train.shape[1:])
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.3, callbacks=get_callbacks())

    show_training_history(history)
    evaluate(model, x_test, y_test)

    prediction = model.predict(x_test, batch_size=BATCH_SIZE)
    glimpse_prediction(x_test, prediction)

    model.save('final_model.h5')

def load_model_and_predict(model_path):
    (_, _), (x_test, _) = datasets.cifar10.load_data()
    model = models.load_model(model_path)

    idx = np.random.randint(0, len(x_test))
    test_image = x_test[idx]
    test_image = test_image.reshape(1, x_test[idx].shape[0], x_test[idx].shape[1], x_test[idx].shape[2])

    prediction = model.predict(test_image)
    show_prediction(x_test[idx], prediction)

if __name__ == '__main__':
    if not IS_TRAINED:
        run()
    else:
        load_model_and_predict('final_model_80.h5')