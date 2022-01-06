import os, re, string

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses

# model = tf.keras.models.load_model('saved_models/text classification')

model = tf.keras.models.load_model('Basic_Text_Classification\text_classification.h5')

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


# New export model
model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])

model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

loss, accuracy = model.evaluate(raw_test_ds)
print(accuracy)

# Test on new data
examples = [
    "Best of the best",
    "Haven't watched a better movie in my life, but it still lacks some creativity",
    "Normally, I would just say it was bad, but this is below everything.",
    "Oh the misery, everybody wants to be my enemy, spare the sympathy, everybody wants to be my enemy"
]

print(model.predict(examples))
