from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import keras
import numpy as np
from keras.models import save_model, load_model
from batch_generator import BatchGenerator
import os
import imageio
from auxilliary import crawl_directory

path = '/home/demokritoscil/Desktop/dftimages_L'
# new = crawl_directory(path)
# list_of_filenames = os.listdir(path)

# list_of_data_names = []
# for filename in list_of_filenames:
#     name,extension = filename.split('.') 
#     list_of_data_names.append(name)

# labels = {}
# for name in list_of_data_names:
#     _,label = name.split('action_')
#     label = label[:-2]
#     labels[name] = int(label)
# print(new[0])
# print(list_of_data_names[0])

genarate_batches = BatchGenerator(path, batch_size=128, dimension=(159,75), nchannels=4,
                 nclasses=51, is_labeled=False)
one_batch_X  = genarate_batches.__getitem__()
print(one_batch_X.shape)

