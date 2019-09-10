# from __future__ import print_function
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# import keras
# import numpy as np
# from keras.models import save_model, load_model
# from batch_generator import BatchGenerator
# import os
# import imageio
# from auxilliary import crawl_directory

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

# genarate_batches = BatchGenerator(path, batch_size=128, dimension=(159,75), nchannels=4,
#                  nclasses=51, is_labeled=False)
# one_batch_X  = genarate_batches.__getitem__()
# print(one_batch_X.shape)

import warnings
warnings.simplefilter(action='ignore')#, category=FutureWarning)
import os
import numpy as np
from keras.datasets import mnist
from keras.models import load_model
import keras as K
from adaptation import AdaPy

(x_train, y_train), (x_test, y_test) = mnist.load_data()

source_representer = load_model('M_s.h5')
source_classifier = load_model('Class.h5')
adda = AdaPy(source_representer,source_classifier)
