# Task: determine the topic of the question
# https://www.tensorflow.org/tutorials/keras/text_classification#exercise_multi-class_classification_on_stack_overflow_questions

import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses

# Stack Overflow dataset
# Balanced sets: 25k for training and testing
url = "http://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"
dataset = tf.keras.utils.get_file("Basic_Text_Classification/stack_overflow_16k.zip", url,
                                  untar=True, cache_dir='.',
                                  cache_subdir='')

# Dataset folder
dataset_dir = os.path.join(os.path.dirname(dataset), 'stack_overflow_16k')



os.listdir(dataset_dir)