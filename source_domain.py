#imports
from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import keras
import numpy as np
from keras.models import save_model, load_model
from batch_generator import DataGenerator
import os
import imageio


# Datasets

path = '/home/demokritoscil/Desktop/dftimages_L/' #string
list_of_filenames = os.listdir(path)

list_of_data_names = []
for filename in list_of_filenames:
  name,extension = filename.split('.') 
  list_of_data_names.append(name)

labels = {}
for name in list_of_data_names:
  _,label = name.split('action_')
  label = label[:-2]
  labels[name] = int(label)


# Batch Generators
genarate_batches = BatchGenerator(list_of_data_names, labels, batch_size=128, dim=(159,75), n_channels=4,
                 n_classes=51, shuffle=True)
one_batch_X , one_batch_y = genarate_batches.__getitem__(path, extension)
print(one_batch_X.shape)

