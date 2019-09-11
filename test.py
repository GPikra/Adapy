#!/usr/bin/env python
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
from auxilliary import one_hot
import h5py

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)* 1/255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)* 1/255
digits = [0,1,2,3,4,5,6,7,8,9]
y_train = one_hot(digits, y_train)
y_test = one_hot(digits, y_test)

i = 0
j = 14
with h5py.File('usps.h5', 'r') as hf:
    train = hf.get('train')
    preX_tr = np.array(train.get('data')[:])
    preX_tr = preX_tr.reshape((preX_tr.shape[0], 16, 16, 1)) 
    X_tr = np.zeros((preX_tr.shape[0], 28, 28, 1)) 
    X_tr[:, 6:22, 6:22] = preX_tr.copy()
    X_tr[:, i:j, i:j] = np.ones((X_tr[:, i:j, i:j].shape[0], j-i, j-i, 1))
    y_tr = train.get('target')[:]
    source = [0,1,2,3,4,5,6,7,8,9]
    y_tr = one_hot(source, y_tr)

    test = hf.get('test')
    preX_te = np.array(test.get('data')[:])
    preX_te = preX_te.reshape((preX_te.shape[0], 16, 16, 1)) 
    X_te = np.zeros((preX_te.shape[0], 28, 28, 1)) 
    X_te[:, 6:22, 6:22] = preX_te.copy()
    X_te[:, i:j, i:j] = np.ones((X_te[:, i:j, i:j].shape[0], j-i, j-i, 1))
    y_te = test.get('target')[:]
    y_te = one_hot(source, y_te)


source_representer = load_model('M_s.h5') #load_model('source.h5') 
source_classifier = load_model('Class.h5')

adda = AdaPy(source_representer,source_classifier, algorithm="wadda", discriminator_per_representer_iterations=25)
pre_transfer_accuracy = adda.target_model.evaluate(X_te, y_te)

adda.fit(X_te,x_train)
post_transfer_accuracy = adda.target_model.evaluate(X_te, y_te)
print("Before", pre_transfer_accuracy)
print("After", post_transfer_accuracy)