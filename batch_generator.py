import keras
import imageio
import random
import numpy as np

from auxilliary import crawl_directory, get_class

#https://github.com/afshinea/keras-data-generator

class BatchGenerator(keras.utils.Sequence):

  def __init__(self, target_directory, batch_size, dimension, nchannels,
                 nclasses, shuffle=True, is_labeled=True):
    """
    #TODO: explain arguments
    #TODO: explain labels {}
    """

    self.__dimension_input = dimension
    self.__batch_size = batch_size
    self.__is_labeled = is_labeled
    self.__target_directory = target_directory
    self.__nchannels = nchannels
    self.__nclasses = nclasses
    self.__shuffle = shuffle

    self.__crawled_directory = crawl_directory(self.__target_directory)
    self.__datasetSize = len(self.__crawled_directory)

    self.on_epoch_end()

    
  def __getitem__(self, path, extension, index=-1):
    """
    Generate one batch of data
    """

    if index == -1:
      upto = (self.__datasetSize//self.__batch_size)-1
      index = randint(0, upto)
    
    indices = self.__indices[index*self.__batch_size:(index+1)*self.__batch_size] 
    list_of_batch_files = [self.__crawled_directory[i] for i in indices]
    X, y = self.__data_generation(list_of_batch_files)

    return X, y

    
  def on_epoch_end(self):
    """
    Shuffle indices after each epoch
    """
    self.__indices = np.arange(self.__datasetSize)
    if self.shuffle == True:
      np.random.shuffle(self.__indices)

  def __data_generation(self, list_of_batch_files):
    """
    Generates data containing batch_size samples: X(n_samples, *dim, n_channels)
    """

    X = np.empty((self.batch_size, *self.dim, self.n_channels))
    y = np.empty((self.batch_size), dtype=np.int8)
    for i, filename in enumerate(list_of_batch_files):
      X[i], y[i] = read_image(filename, self.__is_labeled)
    return X, y