import numpy as np
import keras
import imageio
from random import *

#https://github.com/afshinea/keras-data-generator

class BatchGenerator(keras.utils.Sequence):

  def __init__(self, target_directory, labels={}, batch_size=128, dim=(159,75), n_channels=4,
                 n_classes=51, shuffle=True):
    """
    #TODO: explain arguments
    """

    self.dim = dim
    self.batch_size = batch_size
    self.labels = labels
    self.list_of_data_names = list_of_data_names
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.shuffle = shuffle
    self.on_epoch_end()

    
  def __getitem__(self, path, extension, index=-1):
    """
    Generate one batch of data
    """
    if index == -1:
      upto = (len(self.list_of_data_names)//self.batch_size)-1
      index = randint(0, upto)
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size] 
    list_of_data_names_temp = [self.list_of_data_names[k] for k in indexes]
    X, y = self.__data_generation(list_of_data_names_temp ,path, extension)
    return X, y

    
  def on_epoch_end(self):
    """
    Updates indexes after each epoch
    """

    self.indexes = np.arange(len(self.list_of_data_names))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __data_generation(self, list_of_data_names_temp, path, extension):
    """
    Generates data containing batch_size samples: X(n_samples, *dim, n_channels)
    """

    X = np.empty((self.batch_size, *self.dim, self.n_channels))
    y = np.empty((self.batch_size), dtype=int)

    for i, ID in enumerate(list_of_data_names_temp):
      temp = imageio.imread(path + list_of_data_names_temp[i] + '.' + extension)
      X[i,] = np.array(temp)
      if labels!= {}:
        y[i] = self.labels[ID]
    if lables == {}:
      return X
    else :
      return X, y

