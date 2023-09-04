# Necessary Imports
import cv2
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
import os
# View an image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator


device = 'cuda' if tf.test.is_gpu_available() else 'cpu'


# Path to Kaggle Input
path = "C:/Users/tumul/Documents/GitHub/monument-prediction/Indian-monuments/images"
# Walk through the directory and list number of files
for dirpath, dirnames, filenames in os.walk(path):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


# append the training and the testing paths to the original path
train_dir =  path + "/train/"
test_dir = path + "/test/"

# get all the class names
data_dir = pathlib.Path(train_dir)
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))

# Rescale the data and create data generator instances
train_datagen = ImageDataGenerator(rescale=1/255,)
test_datagen = ImageDataGenerator(rescale=1/255,)
                              

# Load data in from directories and turn it into batches
train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(300, 300),
                                               batch_size=16,
                                               class_mode='categorical') 

test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=(300, 300),
                                             batch_size=16,
                                             class_mode='categorical')

