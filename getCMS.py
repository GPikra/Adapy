#imports
from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from ast import literal_eval
import csv
import keras
from keras.layers import MaxPooling2D, Conv2D, Input, Dense, Reshape, Flatten, Dropout, Embedding, multiply, BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model, model_from_yaml
from keras import optimizers
from keras.datasets import mnist
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from keras.callbacks import TensorBoard
import sys
import numpy as np
from keras.backend import manual_variable_initialization 
from keras.models import save_model, load_model
from create_source import one_hot
from batch_generator import BatchGenerator
# so that load model works properly
manual_variable_initialization(True)

path = '/home/demokritoscil/Desktop/dftimages_M_inputs/'
genarate_batches = BatchGenerator(path, batch_size=600, dimension=(159,75), nchannels=4,
                                    nclasses=51, is_labeled=True)
x_train,y_train  = genarate_batches.get_batch()
path = '/home/demokritoscil/Desktop/dftimages_L_inputs/'
genarate_batches = BatchGenerator(path, batch_size=600, dimension=(159,75), nchannels=4,
                                    nclasses=51, is_labeled=True)
x_test,y_test  = genarate_batches.get_batch()

x_train = x_train.reshape(x_train.shape[0], 159, 75, 4)* 1/255
x_test = x_test.reshape(x_test.shape[0], 159, 75, 4)* 1/255
    
y_train = one_hot(range(1,52), y_train)
y_test = one_hot(range(1,52), y_test)
    

# Define classifier and source representation network

# Load trained source
source_net = load_model('Models/source.h5')

# Define representer for source by chopping classifier from source
M_s = Model(source_net.inputs, source_net.layers[-3].get_output_at(-1))
M_s.summary()
M_s.save('Models/M_s.h5')

M_s = load_model('Models/M_s.h5')
# Define Classifier from rest
inputC = Input((64,))
classifier1 = source_net.layers[-2](inputC) 
classifier2 = source_net.layers[-1](classifier1) 
C = Model(inputC, classifier2)
C.summary()
C.save('Models/Class.h5')

C = load_model('Models/Class.h5')

s_n = Model(M_s.input, C(M_s(M_s.input)))
s_n.summary()

# define optimizer
optimizer = optimizers.Adam()

s_n.compile(loss = 'categorical_crossentropy', optimizer = optimizer,
 metrics = ['accuracy'])

source_net.compile(loss = 'categorical_crossentropy', optimizer = optimizer,
 metrics = ['accuracy'])

# Save models


print( source_net.evaluate(x_test, y_test))
print( s_n.evaluate(x_test, y_test))