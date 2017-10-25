""" This code demonstrates reading the test data and writing 
predictions to an output file.

It should be run from the command line, with one argument:

$ python predict.py [test_file]

where test_file is a .npy file with an identical format to those 
produced by extract_cats_dogs.py for training and validation.

(To test this script, you can use one of those).

This script will create an output file in the same directory 
where it's run, called "predictions.txt".

"""
import sys
import numpy as np
import random
import os
import keras
from keras.models import load_model

CAT_OUTPUT_LABEL = 1
DOG_OUTPUT_LABEL = 0

TEST_FILE = sys.argv[1]
modelDIR =  'keras_cifar10_trained_model.h5'
data = np.load(TEST_FILE).item()
SIZE = 0.01
BATCH =32

images = data["images"]

ids = data["labels"]


OUT_FILE = "predictions.txt"
X = images[:int(len(images)*SIZE)]

print("Writing Predictions to File after Prediction")
out = open(OUT_FILE, "w")
for i, image in enumerate(images):

  image_id = ids[i]
  model = load_model(modelDIR)
  prediction = model.predict(X, BATCH)

  line = str(image_id) + " " + str(prediction) + "\n"
  out.write(line)

out.close()
print("Predictions are written on prediction.txt")