import os

def crawl_directory(dir): 
  subdirs = [x[0] for x in os.walk(dir)]                                                                                               
  tree = []                                                                                                            
  for subdir in subdirs:                                                                                            
    files = next(os.walk(subdir))[2]                                                                             
    if (len(files) > 0):                                                                                          
      for _file in files:                                                                                        
        tree.append(subdir + "/" + _file)                                                                         
  return tree
