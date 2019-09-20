import keras
import random
import numpy as np

from auxilliary import crawl_directory, get_class, read_image, map_labels

#https://github.com/afshinea/keras-data-generator

class BatchGenerator(keras.utils.Sequence):

  def __init__(self, target_directory, batch_size, input_shape,
                 nclasses, shuffle=True, is_labeled=True):
    """
    target_directory : absolute or relative path of the folder, where data files exist in
    batch_size       : number of samples each batch consist of
    input_shape      : dimension and number of channels of the input data
    nclasses         : number of classes to categorise data
    shuffle          : boolean that indicates if indices of data should be shuffled or not
    is_labeled       : boolean that shows if labels exist or not  
    """

    self.__dimension_input = input_shape[:-1]
    self.__target_directory = target_directory
    self.__crawled_directory = sorted(crawl_directory(self.__target_directory))
    self.__datasetSize = len(self.__crawled_directory)

    if batch_size == -1:
      batch_size = self.__datasetSize
    
    self.__batch_size = batch_size
    self.__is_labeled = is_labeled
    self.__nchannels = input_shape[-1]
    self.__nclasses = nclasses
    self.__shuffle = shuffle    
    if self.__is_labeled:
      self.__labels = map_labels(self.__crawled_directory)
    else:
      self.__labels = {}
    self.on_epoch_end()

    
  def __getitem__(self, index):
    """
    Generate one batch of data
    """
    
    if index == -1:
      upto = (self.__datasetSize//self.__batch_size)-1
      index = random.randint(0, upto)
    
    indices = self.__indices[index*self.__batch_size:(index+1)*self.__batch_size] 
    list_of_batch_files = [self.__crawled_directory[i] for i in indices]
    X, y = self.__data_generation(list_of_batch_files)
    if self.__is_labeled:
      return X, y
    return X, -1

    
  def on_epoch_end(self):
    """
    Shuffle indices after each epoch
    """

    self.__indices = np.arange(self.__datasetSize)
    if self.__shuffle == True:
      np.random.shuffle(self.__indices)

  def __data_generation(self, list_of_batch_files):
    """
    Generates data containing batch_size samples: X(n_samples, *dimension_input, nchannels)
    """

    X = np.empty((self.__batch_size, *self.__dimension_input, self.__nchannels))
    y = np.empty((self.__batch_size), dtype=np.int8)
    for i, filename in enumerate(list_of_batch_files):
      X[i], y[i] = read_image(filename, self.__labels, self.__is_labeled)
    return X, y

  def get_batch(self, index=-1):
    return self.__getitem__(index)


class BatchGenerator_Numpy():

  def __init__(self, data, batch_size, shuffle=True, is_labeled = []):
    """
    data       : numpy array of data
    batch_size : number of samples each batch consist of
    shuffle    : boolean that indicates if indices of data should be shuffled or not
    """

    assert isinstance(is_labeled, list), "'is_labeled' must be a list of labels (possibly empty)"
    assert isinstance(data, np.ndarray), "Data must be a numpy array in 'BatchGenerator_numpy'"
    self.__data = data
    self.__shuffle = shuffle
    self.__is_labeled = is_labeled
    self.__datasetSize = self.__data.shape[0]
    if batch_size == -1:
      batch_size = self.__datasetSize
    self.__batch_size = batch_size


  def __getitem__(self, index):
   
    if index == -1:
      upto = (self.__datasetSize//self.__batch_size)-1
      index = random.randint(0, upto)

    self.__indices = np.arange(self.__datasetSize)
    if self.__shuffle == True:
      np.random.shuffle(self.__indices)

    indices = self.__indices[index*self.__batch_size:(index+1)*self.__batch_size]
    if self.__is_labeled:
      return np.array([self.__data[i] for i in indices]), self.__is_labeled[indices]
    return np.array([self.__data[i] for i in indices]), -1

  def get_batch(self, index=-1):
    return self.__getitem__(index)
