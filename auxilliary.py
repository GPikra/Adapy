import os
import imageio

def crawl_directory(dir): 
  subdirs = [x[0] for x in os.walk(dir)]                                                                                               
  tree = []                                                                                                            
  for subdir in subdirs:                                                                                            
    files = next(os.walk(subdir))[2]                                                                             
    if (len(files) > 0):                                                                                          
      for _file in files:                                                                                        
        tree.append(subdir + "/" + _file)                                                                         
  #TODO: Make generator?
  return tree

def get_class(path):
  absolute_subdirectory = path[:path.rfind("/")]
  return absolute_subdirectory[absolute_subdirectory.rfind("/")+1:]

def read_image(path, is_labeled=True):
  read_image = imageio.imread(path)
  if is_labeled:
    label = get_class(path)
    return read_image, label
  return read_image, -1