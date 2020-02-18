# https://www.tensorflow.org/tutorials/images/classification

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from MyUtils import MyUtils
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

TRAIN_DIR = 'train'
TEST_DIR = 'test'
MODEL_DIR = 'model'
LOGS_DIR = 'logs'
BATCH_SIZE = 39
EPOCHS = 100
IMG_HEIGHT = 150
IMG_WIDTH = 150
DEBUG = True

# DATAGEN SETUP

train_datagen = ImageDataGenerator(rescale=1./255)

train_data_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_WIDTH, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='binary'
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_data_gen = test_datagen.flow_from_directory(
    TEST_DIR, target_size=(IMG_WIDTH, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='binary'
)

if DEBUG:
    # display training images
    sample_training_images, _ = next(train_data_gen)
    plt.figure(figsize=(IMG_HEIGHT, IMG_WIDTH))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(sample_training_images[i], cmap=plt.cm.binary)
        # plt.xlabel(class_names[train_labels[i]])
    plt.show()


# create ai model
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# view all the layers of the network using the model's summary method:
if DEBUG:
    model.summary()

total_train = len(os.listdir(TRAIN_DIR))
total_val = len(os.listdir(TEST_DIR))

print(total_train)
print(total_val)

# START TRAINING HELLLA YEEEAGH!
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=test_data_gen,
    validation_steps=total_val // BATCH_SIZE
)

if DEBUG:
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


# EXPORT H5 NODES
MyUtils.maybeMakeDir(MODEL_DIR);
model.save(MODEL_DIR + '/keras-model.h5')


# CONVERT H5 TO TENSORFLOW
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(MODEL_DIR + "/tensorflow-model.tflite", "wb").write(tflite_model)


# VALIDATE TENSORFLOW CONVERSION
