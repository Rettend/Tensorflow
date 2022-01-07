# Regression problem: predict the output of a continuous value, like a price or a probability.
# Task: predict the fuel efficiency (MPG: miles per gallon) of the late-1970s and early 1980s automobiles
# https://www.tensorflow.org/tutorials/keras/regression

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()

# print(dataset.tail()) # Print last 5 rows
# print(dataset.isna().sum()) # Print the Sum of empty data per column

dataset = dataset.dropna() # Drop missing values
# Make the categorical 'Origin' column numerical
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
# print(dataset.tail())

# Split the data into train and test sets (80% train, 20% test)
train_dataset = dataset.sample(frac=0.8, random_state=0) # Retrive random samples
test_dataset = dataset.drop(train_dataset.index) # Remainer of train_dataset goes into test_dataset

sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
# print(train_dataset.describe().transpose())