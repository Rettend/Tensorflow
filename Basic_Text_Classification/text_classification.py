# Task: determine whether the reviews are positive or negative
# Binary classification
# https://www.tensorflow.org/tutorials/keras/text_classification
# https://www.tensorflow.org/text/guide/word_embeddings

from tensorflow.keras import losses
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
# Test GPU, this should return 1
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

relative_path = "Tensorflow/Basic_Text_Classification/"

# Movie reviews dataset
# Balanced sets: 25k for training and 25k for testing
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
                                  untar=True, cache_dir=relative_path + '.',
                                  cache_subdir='')

# Dataset folder
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
# Content: ['README', 'train', 'imdb.vocab', 'imdbEr.txt', 'test']
os.listdir(dataset_dir)

# Train folder
train_dir = os.path.join(dataset_dir, 'train')
# Content: ['unsupBow.feat', 'unsup', 'neg', 'pos', 'urls_pos.txt', 'urls_neg.txt', 'urls_unsup.txt', 'labeledBow.feat']
# The aclImdb/train/pos and aclImdb/train/neg directories contain the positive and negative text files
os.listdir(train_dir)

# # Print one file:
# sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
# with open(sample_file) as f:
#   print(f.read())

# Remove additional folders
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)


# Divide the dataset into 3 splits: Train, Validation, Test
# And load the dataset with tf.keras.utils.text_dataset_from_directory which creates a tf.data.Dataset
# from text files in a directory. It expects the following structure:
# main_directory/
# ...class_a/
# ......a_text_1.txt
# ......a_text_2.txt
# ...class_b/
# ......b_text_1.txt
# ......b_text_2.txt

# Create train set
validation_split = 0.2  # fraction of data to reserve for validation: 20k for training
batch_size = 32  # default
seed = 42  # random seed for shuffling data
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    relative_path + 'aclImdb/train',
    batch_size=batch_size,
    validation_split=validation_split,
    subset='training',
    seed=seed)

# # Print some examples, raw text with HTMK <br/> tags, reviews are labeled 0 (pos) or 1 (neg)
# for text_batch, label_batch in raw_train_ds.take(1):
#   for i in range(3):
#     print("Review", text_batch.numpy()[i])
#     print("Label", label_batch.numpy()[i])

# # Check the class_names for the reviews
# print("Label 0 corresponds to", raw_train_ds.class_names[0])
# print("Label 1 corresponds to", raw_train_ds.class_names[1])

# Create validation set
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    relative_path + 'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

# Create test set
raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    relative_path + 'aclImdb/test',
    batch_size=batch_size)


# Prepare the dataset for training with tf.keras.layers.TextVectorization
# Standardization: preprocessing the text, like removing punctuations and HTML elements, and converting to lower case
# Tokenization: splitting strings into tokens (like splitting sentences to words)
# Vectorization: converting tokens into numbers for the neural network, it can be done inside or outside the model
# using it outside lets it to be run on GPU, while using it inside enables the model to be exported and prevents train/test skew
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)  # lowercase strings
    stripped_html = tf.strings.regex_replace(
        lowercase, '<br />', ' ')  # remove HTML
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


# Create layer
max_features = 10000
sequence_length = 250
vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

# Function to see the results


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# # retrieve a batch (of 32 reviews and labels) from the dataset
# text_batch, label_batch = next(iter(raw_train_ds))
# first_review, first_label = text_batch[0], label_batch[0]
# print("Review", first_review)
# print("Label", raw_train_ds.class_names[first_label])
# print("Vectorized review", vectorize_text(first_review, first_label))

# # Lookup specific integers and what they correspond to
# print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
# print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
# print("   3 ---> ",vectorize_layer.get_vocabulary()[3])
# print("   2 ---> ",vectorize_layer.get_vocabulary()[2])
# print("   1 ---> ",vectorize_layer.get_vocabulary()[1]) # space?
# print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

# Apply the TextVectorization layer to the sets
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Configure the dataset for performance
# .cache() keeps the data in memory after it's loaded off disk
# .prefetch() overlaps data preprocessing and model execution while training
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Create the model
embedding_dim = 16
model = tf.keras.Sequential([
    # Takes the integer-encoded reviews and looks up an embedding vector for each word-index
    layers.Embedding(max_features + 1, embedding_dim),  # 16 output nodes
    layers.Dropout(0.2),  # Randomly sets inputs to 0, reduce overfitting
    layers.GlobalAveragePooling1D(),  # Average pooling for data
    # 0.2 is the rate at which the random selection happens
    layers.Dropout(0.2),
    layers.Dense(1)])  # densely connected single output layer with 16 hidden units

model.summary()

# Loss function and optimizer
# BinaryCrossentropy is used for binary classification where the output is a probability
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

# Train the model
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

# Evaluation
loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

# Plot the accuracy and loss over time
history_dict = history.history
# ['loss', 'binary_accuracy', 'val_loss', 'val_binary_accuracy']
# history_dict.keys()

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'c', label='Validition loss')
plt.title('Training and validition loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'c', label='Validation acc')
plt.title('Training and validition accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()
# !Danger Overfitting!

# Export the model
