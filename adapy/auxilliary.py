import os
import imageio
import numpy as np
from keras.models import clone_model
import keras.backend as K
from keras.constraints import Constraint
import cv2 as cv

class WeightClip(Constraint):
  """
  Keras weight constraint for clipping weights. Used for enforcing Lipschitz condition.
  """

  def __init__(self, clip_parameter=0.01):
      self.clip_parameter = clip_parameter

  def __call__(self, p):
      return K.clip(p, -self.clip_parameter, self.clip_parameter)

  def get_config(self):
      return {'name': self.__class__.__name__,
              'clip_parameter': self.clip_parameter}


def entropy_loss(yTrue,yPred):
  """
  Entropy loss function for keras.
  """

  return -1*K.mean(yPred*K.log(yPred))


def wasserstein_loss(y_true, y_pred):
  """
  Keras loss function for Wasserstein Domain Critic.
  
  Label target samples -1 and source samples 1
  """

  return -K.mean(y_true * y_pred)


def copy_model(model, model_name):
  "Safe copy keras model function"

  result_model = clone_model(model)
  result_model.set_weights(model.get_weights()) 
  result_model.name = model_name
  return result_model


def crawl_directory(dir): 
  """
  Function for listing all files in the directory tree with root (input):"dir"
  """

  subdirs = [x[0] for x in os.walk(dir)]                                                                                               
  tree = []                                                                                                            
  for subdir in subdirs:                                                                                            
    files = next(os.walk(subdir))[2]                                                                             
    if (len(files) > 0):                                                                                          
      for _file in files:                                                                                        
        tree.append(subdir + "/" + _file)                                                                         
  #TODO: Make generator -> problems: cannot have dataSize , cannot have indexing in labels 
  return tree


def get_class(path):
  """
  Gets the class of a given file (name of the subdirectory it is in)
  """

  absolute_subdirectory = path[:path.rfind("/")]
  return absolute_subdirectory[absolute_subdirectory.rfind("/")+1:]

#TODO: Add arguments to handle different channel formats (RGB, BGR etc)
def read_image(path, mapped_labels, is_labeled=True):
  read_image = cv.imread(path)
  if is_labeled:
    label = get_class(path)
    return read_image, mapped_labels[label]
  return read_image, -1


def map_labels(directories):
  """
  Indexing of labels from strings to integers
  """

  labels = set()
  for path in directories:
    labels.add(get_class(path))
  mapped_labels = {}
  i = 0
  for label in labels:
    mapped_labels[label] = i
    i+=1
  return mapped_labels


def one_hot(data_labels, labels):
  """
  Converts labels to one-hot encodings

    - labels: list, list of labels to attempt to convert
    - data_labels: list, list of labels in dataset
  """

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

def hot_one(labels, data_labels):
    aux = []
    for i in data_labels:
        aux.append(labels[np.argmax(i)])
    return aux
