""" This code demostrates reading the training and validation data
produced by extract_cats_dogs.py

You will need to have Pillow installed to display the images.

http://pillow.readthedocs.io/en/3.4.x/installation.html
"""

import numpy as np
from PIL import Image

TRAIN_PATH = "train.npy"
VAL_PATH = "validation.npy"

CAT_OUTPUT_LABEL = 1
DOG_OUTPUT_LABEL = 0

def load(npy_file):
  data = np.load(npy_file).item()
  return data['images'], data['labels']

train_images, train_labels = load(TRAIN_PATH)
val_images, val_labels = load(VAL_PATH)

# Make sure the images look correct
i = 0
image = train_images[i]
label = train_labels[i]
if label == CAT_OUTPUT_LABEL: 
  print ("Cat!")
else:
  print ("Dog!")

im = Image.fromarray(image)
im.show()