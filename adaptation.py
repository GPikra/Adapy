import warnings
warnings.simplefilter(action='ignore')#, category=FutureWarning)
import os
import numpy as np

import keras as K
import keras.layers as l
import keras.optimizers as o
from keras.models import clone_model, Model


from batch_generator import DataGenerator

class AdaPy():

  def __init__(self, source_representer, source_classifier, index_to_label_dictionary = None, algorithm="adda", domain_discriminator = "linear"):
    assert algorithm in ["adda", "wadda"], "Invalid choice of algorithm"
    assert isinstance(source_representer, K.engine.training.Model) and isinstance(source_classifier, K.engine.training.Model), \
    "Provide keras models for source encoder and classifier"

    self.__latent_dimensions = source_representer.output_shape
    self.__shape = source_representer.input_shape
    self.__nlabels = source_classifier.output_shape

    self.__algorithm = algorithm

    self.__source_representer = clone_model(source_representer)
    self.__source_representer.set_weights(source_representer.get_weights()) 

    self.__target_representer = clone_model(source_representer)
    self.__target_representer.set_weights(source_representer.get_weights()) 

    if domain_discriminator == "linear":
      self.__domain_discriminator = self.__build_domain_discriminator()
    else:
      assert isinstance(domain_discriminator, K.engine.training.Model), "Provide keras model for domain discriminator"
      assert domain_discriminator.output_shape == self.__nlabels, "Domain discriminator must be a binary classifier"
      assert domain_discriminator.input_shape == self.__latent_dimensions, "Domain discriminator input dimensionality was invalid"
      self.__domain_discriminator = domain_discriminator

    self.__source_classifier = clone_model(source_classifier)
    self.__source_classifier.set_weights(source_classifier.get_weights()) 


    # representer_input = l.Input(self.__shape)

    # source_representer_output = self.__source_representer(representer_input)
    # source_classifier = self.__source_classifier(source_representer_output)
    # self.__source_model = Model(representer_input, source_classifier)





  def __build_domain_discriminator(self):
    if self.__algorithm == "adda":
      latent_representation = l.Input(self.__latent_dimensions)
      classifier = l.Dense(1, activation="sigmoid")(latent_representation)
      domain_discriminator = Model(latent_representation, classifier)
      domain_discriminator.compile(loss="binary_crossentropy", optimizer = o.Adam())
    return domain_discriminator



  @property
  def domain_discriminator(self):
    return self.__domain_discriminator

  @property
  def nlabels(self):
    return self.__nlabels

  @property
  def latent_dimensions(self):
    return self.__latent_dimensions

  @property
  def input_shape(self):
    return self.__shape[1:]

  @property
  def source_classifier(self):
    return self.__source_classifier

  @property
  def source_representer(self):
    return self.__source_representer

  @property
  def source_classifier_summary(self):
    self.__source_classifier.summary()

  @property
  def source_representer_summary(self):
    self.__source_representer.summary()

  @property
  def source_model(self):
    return self.__source_model

  @property
  def source_model_summary(self):
    self.__source_model.summary()

  @property
  def target_data(self):
    return self.__target_data

  @target_data.setter
  def target_data(self, value):
    if isinstance(np.array((0,1)), str):
      assert os.path.exists(value)




    if isinstance(value, np.ndarray):
      assert value.shape == self.__shape
      self.__target_data = value