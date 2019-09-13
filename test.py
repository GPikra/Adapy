#!/usr/bin/env python

import warnings
warnings.simplefilter(action='ignore')#, category=FutureWarning)
import os
import numpy as np
from keras.datasets import mnist
from keras.models import load_model
import keras as K
from adaptation import AdaPy
from auxilliary import one_hot
from batch_generator import BatchGenerator
import h5py

# # (x_train, y_train), (x_test, y_test) = mnist.load_data()
# # x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)* 1/255
# # x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)* 1/255
# # digits = [0,1,2,3,4,5,6,7,8,9]
# # y_train = one_hot(digits, y_train)
# # y_test = one_hot(digits, y_test)

# # i = 0
# # j = 14
# # with h5py.File('usps.h5', 'r') as hf:
# #     train = hf.get('train')
# #     preX_tr = np.array(train.get('data')[:])
# #     preX_tr = preX_tr.reshape((preX_tr.shape[0], 16, 16, 1)) 
# #     X_tr = np.zeros((preX_tr.shape[0], 28, 28, 1)) 
# #     X_tr[:, 6:22, 6:22] = preX_tr.copy()
# #     X_tr[:, i:j, i:j] = np.ones((X_tr[:, i:j, i:j].shape[0], j-i, j-i, 1))
# #     y_tr = train.get('target')[:]
# #     source = [0,1,2,3,4,5,6,7,8,9]
# #     y_tr = one_hot(source, y_tr)

# #     test = hf.get('test')
# #     preX_te = np.array(test.get('data')[:])
# #     preX_te = preX_te.reshape((preX_te.shape[0], 16, 16, 1)) 
# #     X_te = np.zeros((preX_te.shape[0], 28, 28, 1)) 
# #     X_te[:, 6:22, 6:22] = preX_te.copy()
# #     X_te[:, i:j, i:j] = np.ones((X_te[:, i:j, i:j].shape[0], j-i, j-i, 1))
# #     y_te = test.get('target')[:]
# #     y_te = one_hot(source, y_te)

def one_hot(labels, data_labels):
    labels = list(set(labels))
    labels.sort()
    a = {}
    for i , v in enumerate(labels):
        a[v] = i  
    A = np.zeros((data_labels.shape[0],len(labels)))
    for i in range(A.shape[0]):
        v = data_labels[i]
        if v not in a.keys(): pass
        else: A[i][a[v]] = 1
    return A

source_representer = load_model('../models/image_encoder.h5') 
source_classifier = load_model('../models/classifier.h5')

adda = AdaPy(source_representer,source_classifier)
adda.fit('../emotion_noise/savee_specs','../emotion_noise/german_specs')
