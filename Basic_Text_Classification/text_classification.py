# Task: determine whether the reviews are positive or negative
# https://www.tensorflow.org/tutorials/keras/text_classification

import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
# Test GPU, this should return 1
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras import layers
from tensorflow.keras import losses

# Movie reviews dataset
# Balanced sets: 25k for training and testing
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tf.keras.utils.get_file("Basic_Text_Classification/aclImdb_v1.tar.gz", url,
                                  untar=True, cache_dir='.',
                                  cache_subdir='')

# Dataset folder
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

# Train folder
train_dir = os.path.join(dataset_dir, 'train')

# Remove additional folders
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

# Sets
batch_size = 32
seed = 42
# Traing
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)
# Validation
# Create validation set using 80:20 split from training
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)
# Test
raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test',
    batch_size=batch_size)

# Preprocessing: prepare dataset for testing


def custom_standardization(input_data):
    # Lowercase data
    lowercase = tf.strings.lower(input_data)
    # Remove HTML linebreak tags
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


# TextVertorization: text to numbers
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


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]

# Apply TextVectorization to datasets
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Cache and Prefetch for performance improvement
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Creat Model
# 16 hidden units
embedding_dim = 16

# To prevent Overfitting, this callback will stop the training when there is no improvement in
# the loss for three consecutive epochs.
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

model = tf.keras.Sequential([
    # The Embedding layer takes the integer-encoded reviews and looks up
    # an embedding vector for each word-index
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    # The GlobalAveragePooling1D layer returns a fixed-length output
    # vector for each example by averaging over the sequence dimension
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    # Densely connected layer with a single output node
    layers.Dense(1)])
model.summary()

# Loss function and optimizer
# Adam we do
# losses.BinaryCeossetropy is used for binary classification which
# outputs a probability (a single-unit layer with a sigmoid activation)
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

# Training the Model
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[callback])

# Evaluation
loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

# New export model
export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)

# Test on new data
examples = [
  "Haven't watched a better movie in my life.",
  "Normally, I would just say it was bad, but this is below everything.",
  "Oh the misery, everybody wants to be my enemy, spare the sympathy, everybody wants to be my enemy"
]

print(export_model.predict(examples))

# Evaluation
loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

# Plot the accuracy and loss over time
history_dict = history.history

# 4 entries monitored:
# dict_keys(['loss', 'binary_accuracy', 'val_loss', 'val_binary_accuracy'])
history_dict.keys()

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


# Save the entire model as a SavedModel.
# Access with: model = tf.keras.models.load_model('saved_models/text classification')
#export_model.save('saved_models/text classification')

# HDF5 Saving method
model.save('Basic_Text_Classification/text_classification.h5')