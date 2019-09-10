import os
import imageio
import numpy as np
from keras.models import clone_model

def copy_model(model, model_name):
  result_model = clone_model(model)
  result_model.set_weights(model.get_weights()) 
  result_model.name = model_name
  return result_model

def crawl_directory(dir): 
  subdirs = [x[0] for x in os.walk(dir)]                                                                                               
  tree = []                                                                                                            
  for subdir in subdirs:                                                                                            
    files = next(os.walk(subdir))[2]                                                                             
    if (len(files) > 0):                                                                                          
      for _file in files:                                                                                        
        tree.append(subdir + "/" + _file)                                                                         
  #TODO: Make generator -> problems: cannot have dataSize , cannot have indexing in labels 
  yield tree

def get_class(path):
  absolute_subdirectory = path[:path.rfind("/")]
  return absolute_subdirectory[absolute_subdirectory.rfind("/")+1:]

def read_image(path, mapped_labels, is_labeled=True):
  read_image = imageio.imread(path)
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

def hot_one(labels, data_labels):
    aux = []
    for i in data_labels:
        aux.append(labels[np.argmax(i)])
    return aux

