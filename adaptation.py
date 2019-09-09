import warnings
warnings.simplefilter(action='ignore')#, category=FutureWarning)
import os
import numpy as np

import keras as K
import keras.layers as l
import keras.optimizers as o
from keras.models import clone_model, Model


from batch_generator import BatchGenerator, BatchGenerator_Numpy

class AdaPy():

  def __init__(self,
   source_representer, 
   source_classifier,
   index_to_label_dictionary = None, 
   algorithm="adda", 
   domain_discriminator = "linear", 
   discriminator_lr = 0.001,
   target_representer_lr = 0.0002,
   discriminator_per_representer_iterations = 10,
   batch_size = 32,
   epochs = 10,
   target_dim = (25,25),
   target_nchannels = 3,
   source_dim = (65,65),
   source_nchannels = 3,
   output_directory = "Models/"
   ):
    """
    #TODO:Add argument descriptions
    """

    assert algorithm in ["adda", "wadda"], "Invalid choice of algorithm"
    assert isinstance(source_representer, K.engine.training.Model) and isinstance(source_classifier, K.engine.training.Model), \
    "Provide keras models for source encoder and classifier"

    self.__output_directory = output_directory

    self.__discriminator_learning_rate = discriminator_lr
    self.__target_representer_learning_rate = target_representer_lr

    self.__latent_dimensions = source_representer.output_shape
    self.__shape = source_representer.input_shape
    self.__nlabels = source_classifier.output_shape
    self.__target_data_dimension = target_dim
    self.__number_of_target_data_channels = target_nchannels
    self.__source_data_dimension = source_dim
    self.__number_of_source_data_channels = source_nchannels
    self.__batch_size = batch_size
    self.__epochs = epochs
    self.__discriminator_per_representer_iterations = discriminator_per_representer_iterations

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

    representer_input = l.Input(self.input_shape)

    source_representer_output = self.__source_representer(representer_input)
    source_classifier = self.__source_classifier(source_representer_output)
    self.__source_model = Model(representer_input, source_classifier)

    target_representer_output = self.__target_representer(representer_input)
    domain_discriminator_o_target_representer = self.__domain_discriminator(target_representer_output)
    self.__train_target = Model(representer_input, domain_discriminator_o_target_representer)

    self.compile_models()

  def compile_models(self):
    if self.__algorithm == "adda":
      self.__domain_discriminator.trainable = True
      self.__domain_discriminator.compile(loss="binary_crossentropy", optimizer = o.Adam(lr=self.__discriminator_learning_rate))
      self.__domain_discriminator.trainable = False
      self.__train_target.compile(loss="binary_crossentropy", optimizer = o.Adam(lr=self.__target_representer_learning_rate))


  def __build_domain_discriminator(self):
    if self.__algorithm == "adda":
      latent_representation = l.Input(self.__latent_dimensions)
      classifier = l.Dense(1, activation="sigmoid")(latent_representation)
      domain_discriminator = Model(latent_representation, classifier)
    return domain_discriminator

  #TODO: Xtarget, Xsource, handle arguments
  def fit(self, Xtarget, Xsource, epochs=self.__epochs):
    #TODO:add argument description
    """
    Xtarget : numpy array of target data or absolute/relative path of the folder, where target data files exist in
    Xsource : numpy array of source data or absolute/relative path of the folder, where source data files exist in
    epochs  : number of epochs that model will be trained for
    """

    self.target_data = Xtarget
    self.source_data = Xsource
    if self.__algorithm == 'adda':
      if isinstance(self.target_data, BatchGenerator_Numpy):
        source_label = np.ones((self.__batch_size, 1))
        target_label = np.zeros((self.__batch_size, 1))

        for _ in range(epochs):
          for _ in range(self.__discriminator_per_representer_iterations):
            target_data = self.target_data.get_batch()
            source_data = self.source_data.get_batch()
            #TODO:issue Tensorboard   
            target_latent = self.__target_representer.predict(target_data)
            source_latent = self.__source_representer.predict(source_data)   

            #TODO:Handle source batch differently?
            self.__domain_discriminator.train_on_batch(target_latent, target_label)
            self.__domain_discriminator.train_on_batch(source_latent, source_label)
            self.__train_target.train_on_batch(target_data, source_label)
      
      if isinstance(self.target_data, BatchGenerator):
        #TODO: Add Batchgenerator training functionality
        pass

  @property
  def domain_discriminator_lr(self):
    return self.__discriminator_learning_rate
  

  @domain_discriminator_lr.setter
  def domain_discriminator_lr(self, value):
    self.__discriminator_learning_rate = value


  @property
  def target_lr(self):
    return self.__target_representer_learning_rate
  

  @target_lr.setter
  def target_lr(self, value):
    self.__target_representer_learning_rate = value

  #TODO:Issue
  @property
  def batch_size(self):
    return self.__batch_size
  
  @batch_size.setter
  def batch_size(self, value):
    # assert (value > 0) and (value < self.)
    self.__batch_size = value

  @property
  def epochs(self):
    return self.__epochs
  
  @epochs.setter
  def epochs(self, value):
    assert (value > 0) and isinstance(value, int), "epochs must be a positive integer" 
    self.__epochs = value

  @property
  def shuffle(self):
    return self.__shuffle
  
  @shuffle.setter
  def shuffle(self, value):
    assert isinstance(value, bool), "shuffle must be boolean"
    self.__shuffle = value

  #TODO:add description
  @property
  def dpr(self):
    return self.__discriminator_per_representer_iterations

  @dpr.setter
  def dpr(self,value):
    assert value >= 1, "dpr must be >= 1"
    self.__discriminator_per_representer_iterations = value

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
  def output_directory(self):
    return self.__output_directory

  @output_directory.setter
  def output_directory(self, value):
    assert isinstance(value, str) and os.path.exists(value), "Provide a valid output directory for model storage"
    self.__output_directory = value

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
    if isinstance(value, str):
      assert os.path.exists(value), "Invalid target domain directory"
      #TODO: add arguments to BatchGenerator
      self.__target_generator = BatchGenerator(value, self.__batch_size, self.__target_data_dimension, 
                                                self.__number_of_target_data_channels, self.__nlabels, self.__shuffle, False)

    if isinstance(value, np.ndarray):
      assert value.shape == self.__shape, "Invalid target domain dimensions"
      #TODO: add arguments to BatchGenerator_Numpy
      self.__target_data = BatchGenerator_Numpy(value, self.__batch_size, self.__shuffle)


  @property
  def source_data(self):
    return self.__source_data


  @source_data.setter
  def source_data(self, value):
    if isinstance(value, str):
      assert os.path.exists(value), "Invalid source domain directory"
      #TODO: add arguments to BatchGenerator
      self.__source_generator = BatchGenerator(value,self.__batch_size, self.__source_data_dimension, 
                                                self.__number_of_source_data_channels, self.__nlabels, self.__shuffle)

    if isinstance(value, np.ndarray):
      assert value.shape == self.__shape, "Invalid source domain dimensions"
      #TODO: add arguments to BatchGenerator_Numpy
      self.__source_data = BatchGenerator_Numpy(value, self.__batch_size, self.__shuffle)