# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:19:10 2017

@author: Mohitosh
"""

import numpy as np
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

DATA_FILE = "train.npy"
TEST_PATH = "validation.npy"
DATA_SIZE = 1  #percentage of current dataset to be used (.01 means 1%)
NUM_CLASSES = 2 #dogs or cats
BATCH_SIZE = 32
EPOCHS = 20
SAVE= os.getcwd()
MODEL = 'keras_cifar10_trained_model.h5'

def load(npy_file):
  data = np.load(npy_file).item()
  return data['images'], data['labels']


train_images, train_labels = load(DATA_FILE)
test_images, test_labels = load(TEST_PATH)

# reducing our dataset for testing purposes.
X = train_images[:int(len(train_images)*DATA_SIZE)]
Y = train_labels[:int(len(train_labels)*DATA_SIZE)]

X_test = test_images[:int(len(test_images)*DATA_SIZE)]
Y_test = test_labels[:int(len(test_labels)*DATA_SIZE)]

X = X.astype('float32')
X_test = X_test.astype('float32')
X /= 255
X_test /= 255

Y = keras.utils.to_categorical(Y, NUM_CLASSES)
Y_test = keras.utils.to_categorical(Y_test, NUM_CLASSES)



print("X Shape:",X.shape[1:])

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=X.shape[1:]))

model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

model.fit(X, Y, validation_data=(X_test, Y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)

# Evaluating the trained model
scores = model.evaluate(X_test, Y_test, verbose=0)
print('\nTest loss:', scores[0])
print('Test accuracy:', scores[1]*100)


if not os.path.isdir(SAVE):
    os.makedirs(SAVE)
model_path = os.path.join(SAVE, MODEL)
model.save(model_path)
print('\nSaved trained model at %s ' % model_path)
