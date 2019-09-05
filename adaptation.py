import keras as K
from keras.models import clone_model
import os

class AdaPy():

  def __init__(self, source_representer, source_classifier, algorithm="adda"):
    assert algorithm in ["adda", "wadda"], "Invalid choice of algorithm"
    assert isinstance(source_representer, K.engine.training.Model) and isinstance(source_classifier, K.engine.training.Model), \
    "Provide keras models for source encoder and classifier"

    self.__source_representer = clone_model(source_representer)
    self.__source_representer.set_weights(source_representer.get_weights()) 

    self.__source_classifier = clone_model(source_classifier)
    self.__source_classifier.set_weights(source_classifier.get_weights()) 

    self.__shape = source_representer.input_shape()

  @property
  def target_data(self):
    return self.__target_data

  @target_data.setter
  def target_data(self, value):
    if type(value) == str:
      assert os.path.exists(value)
    if type(value) == numpy.array:
      assert value.shape == self.__shape
    self.__target_data = value