# Regression problem: predict the output of a continuous value, like a price or a probability.
# Task: predict the fuel efficiency (MPG: miles per gallon) of the late-1970s and early 1980s automobiles
# https://www.tensorflow.org/tutorials/keras/regression

from pickletools import optimize
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()

# print(dataset.tail()) # Print last 5 rows
# print(dataset.isna().sum()) # Print the Sum of empty data per column

dataset = dataset.dropna()  # Drop missing values
# Make the categorical 'Origin' column numerical
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
# print(dataset.tail())

# Split the data into train and test sets (80% train, 20% test)
train_dataset = dataset.sample(
    frac=0.8, random_state=0)  # Retrive random samples
# Remainer of train_dataset goes into test_dataset
test_dataset = dataset.drop(train_dataset.index)

sns.pairplot(
    train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
# print(train_dataset.describe().transpose())

# Split features from labels
train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

# Normalization
train_dataset.describe().transpose()[['mean', 'std']]
# Normalization layer
normalizer = tf.keras.layers.Normalization(axis=-1)
# Call adapt
normalizer.adapt(np.array(train_features))

# # Print results
# print(normalizer.mean.numpy())
# first = np.array(train_features[:1])
# with np.printoptions(precision=2, suppress=True):
#   print('First example:', first)
#   print('Normalized:', normalizer(first).numpy())

# Linear Regression: predict 'MPG' from 'Horsepower'
# 1. Normalize 'Horsepower' using the Normalization layer
# 2. Apply linear transformation (y = mx + b) to produce 1 output using a Dense layer
# input_shape will be set automatically
horsepower = np.array(train_features['Horsepower'])
horsepower_normalizer = layers.Normalization(input_shape=[1, ], axis=None)
horsepower_normalizer.adapt(horsepower)
# Sequential model
horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])
horsepower_model.summary()

# print(horsepower_model.predict(horsepower[:10]))

# Loss function and optimizer
horsepower_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

# Run for 100 epochs
history = horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split=0.2)

# Retrive history
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

# Plot


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()


# plot_loss(history)

# Store test set
test_results = {}
test_results['horsepower_model'] = horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels, verbose=0)

x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)


def plot_horsepower(x, y):
    plt.scatter(train_features['Horsepower'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.legend()
    plt.show()

plot_horsepower(x, y)

# Linear regression with multiple inputs
# y = mx + b
# m is a matrix and b is a vector
# new model with the same normalizer
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

# # Calling Model.predict will produce outputs with units=1
# linear_model.predict(train_features[:10]) # take first 10 as inputs
# # array([[ 0.324], [-0.46 ], [-1.102], [ 0.466], [ 1.012], [-1.399], [ 0.817], [-0.906], [-1.116], [ 0.648]], dtype=float32)

# # Look at the weigths with kernel, shape: (9, 1)
# linear_model.layers[1].kernel
# # <tf.Variable 'dense_1/kernel:0' shape=(9, 1) dtype=float32, numpy=array([[-0.148], [-0.714], [ 0.694], [-0.349], [-0.679], [ 0.235], [ 0.448], [ 0.546], [ 0.006]], dtype=float32)>

# Configure the model
linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

# Train for 100 epochs
history = linear_model.fit(
    train_features,
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

# plot_loss(history)

# Collect the results on the test set for later
test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=0)

# Regression with a Deep Neural Network (DNN)
# additionally contains hidden non-linear layers with ReLU activation
# Two models: single-input and multiple-input
# For this the compile method will be the same so we use a function (normalizers differ)
def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(
        loss='mean_absolute_error',
        optimizer=tf.keras.optimizers.Adam(0.001))
    
    return model

# Single-input DNN
# takes 'Horsepower' as the only input and horsepower_normalizer layer
dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)
dnn_horsepower_model.summary()

# Train
history = dnn_horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=100
)

# plot_loss(history)

x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)
plot_horsepower(x, y)