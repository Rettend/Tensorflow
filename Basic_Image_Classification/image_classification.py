# Task: classify images
# https://www.tensorflow.org/tutorials/keras/classification

# https://www.tensorflow.org/tutorials/keras/classification#make_predictions
# https://www.tensorflow.org/tutorials/keras/overfit_and_underfit#demonstrate_overfitting
# https://www.tensorflow.org/tutorials/keras/overfit_and_underfit#strategies_to_prevent_overfitting

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Import Fashon MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

# Training set: 60k images for training the network
# Testing set: 10k images for verifying the network
# Images: 28x28 NumPy arrays, pixel values from 0 to 255
# Labels: integers from 0 to 9, corresponding to the class of the images
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

# Name the 10 image classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Image preprocessing
# Normalizing the image values to range from 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Verify data format
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Layers
model = tf.keras.Sequential([
    # Flatten: Reformats the images to one dimensions, 28 * 28 = 784p
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # Dense: fully connected 128 and 10 node (neuron) layers
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10)

# Test the model on the test dataset
# Overfitting: test accuracy is lower than the training --> The model learns from the noise which negatively impacts performance
# deep learning models tend to be good at fitting to the training data, but the real challenge is generalization, not fitting
# Find a balance between "too much capacity" and "not enough capacity"
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}\nTest loss: {test_loss}')
