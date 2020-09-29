# Modified version of https://www.tensorflow.org/tutorials/generative/dcgan
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend, datasets, models, layers, losses, callbacks
import matplotlib.pyplot as plt
import time

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

backend.set_image_data_format('channels_last')
np.random.seed(42)

BUFFER_SIZE = 60000
BATCH_SIZE = 256
IMAGE_SIZE = 28
NOISE_VECTOR_SIZE = 100
EPOCHS = 100
num_examples_to_generate = 16

SAVED_MODEL_PATH = 'trained_gan.h5'

def make_generator_model():
    model = models.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(NOISE_VECTOR_SIZE,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='relu'))
    assert model.output_shape == (None, IMAGE_SIZE, IMAGE_SIZE, 1)

    return model

def make_discriminator_model():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[IMAGE_SIZE, IMAGE_SIZE, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Fool discriminator to predict 1 for fake images
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss

    return total_loss

def generate_and_show_images(model, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 255.0, cmap='gray')
        plt.axis('off')
    
    plt.savefig('generated.png')
    plt.show()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(images, generator, discriminator):
    noise = tf.random.normal([BATCH_SIZE, NOISE_VECTOR_SIZE])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    #print("Generator loss: %.4f - Discriminator loss: %.4f\n" % (float(gen_loss),float(disc_loss)))

def train(dataset, epochs, generator, discriminator):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch, generator, discriminator)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

def run():
    (x_train, _), (_, _) = datasets.fashion_mnist.load_data()

    x_train = x_train.astype('float32')
    x_train = x_train / 255.0
    x_train = np.reshape(x_train, list(x_train.shape) + [1])

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    generator = make_generator_model()
    discriminator = make_discriminator_model()

    train(train_dataset, EPOCHS, generator, discriminator)

    generator.save(SAVED_MODEL_PATH)
    test_batch = tf.random.normal([num_examples_to_generate, NOISE_VECTOR_SIZE])
    generate_and_show_images(generator, test_batch)

if __name__ == '__main__':
    run()

