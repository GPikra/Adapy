import os
import imageio

class BatchGenerator_numpy():

  def __init__(self, data, batch_size):
    assert isinstance(data, np.ndarray), "Data must be a numpy array in 'BatchGenerator_numpy'"
    pass

  def __get_item__(self):
    pass

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
  labels = set()
  for path in directories:
    labels.add(get_class(path))
  mapped_labels = {}
  i = 0
  for label in labels:
    mapped_labels[label] = i
    i+=1
  return mapped_labels

