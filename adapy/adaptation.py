#!/usr/bin/env python
import warnings
warnings.simplefilter(action='ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
import numpy as np

from tqdm import tqdm

import keras as K
import keras.layers as l
import keras.optimizers as o
from keras.models import Model

from adapy.auxilliary import copy_model, WeightClip, wasserstein_loss, one_hot
from adapy.batch_generator import BatchGenerator, BatchGenerator_Numpy


class AdaPy():
  """
    Simple class for adversarial domain adaptation bases on keras and tensorflow. 

      Input:
        - source_representer                       : model be trained by source data
        - source_classifier                        : model of classifier be trained by source data
        - index_to_label_dictionary                : specific way to index labels              #not sure what you mean
        - algorithm                                : adda or wadda algorithm choice for training   
        - domain_discriminator                     : method to use for domain discriminator #TODO: Add a 1-hidden layer option
        - discriminator_lr                         : learning rate of discriminator
        - target_representer_lr                    : learning rate in training target data
        - discriminator_per_representer_iterations : iterations for discriminator training according to representer training 
        - batch_size                               : number of samples each batch consist of
        - epochs                                   : number of epochs that model will be trained for
        - output_directory                         : directory to save output models
    """

  def __init__(self,
   source_representer, 
   source_classifier,
   index_to_label_dictionary = None, 
   algorithm="adda", 
   domain_discriminator = "linear", 
   discriminator_lr = 0.001,
   target_representer_lr = 0.0002,
   discriminator_per_representer_iterations = 10,
   discriminator_per_representer_iterations_for0 = 25,
   batch_size = 256,
   weight_clip_threshold = 0.05,
   epochs = 5,
   output_directory = "Models/",
   lipschitz = "clip"
   ):

    assert algorithm in ["adda", "wadda"], "Invalid choice of algorithm"
    assert isinstance(source_representer, K.engine.training.Model) and isinstance(source_classifier, K.engine.training.Model), \
    "Provide keras models for source encoder and classifier"
    #TODO: Add assertions for all arguments

    self.__output_directory = output_directory
    self.__algorithm = algorithm
    self.__discriminator_learning_rate = discriminator_lr
    self.__target_representer_learning_rate = target_representer_lr
    self.__weight_clip_threshold = weight_clip_threshold
    self.__shuffle = True
    self.__epochs = epochs
    self.__discriminator_per_representer_iterations = discriminator_per_representer_iterations
    self.__discriminator_per_representer_iterations_for0 = discriminator_per_representer_iterations_for0
    self.__batch_size = batch_size
    self.__latent_dimensions = source_representer.output_shape[1]
    self.__shape = source_representer.input_shape
    self.__nlabels = source_classifier.output_shape[-1]
    if index_to_label_dictionary is None:
      self.__index_to_label_dictionary = {k:"" for k in range(self.__nlabels)}
    else:
      self.__index_to_label_dictionary = index_to_label_dictionary
    self.__initialize_models(source_representer, source_classifier, domain_discriminator)
    self.__define_models_for_training_and_inference()
    self.compile_models()

  def compile_models(self):
    """
    Method to compile all models of object
    """
    
    if self.__algorithm == "adda":
      self.__domain_discriminator.trainable = True
      self.__domain_discriminator.compile(loss="binary_crossentropy", optimizer = o.Adam(lr=self.__discriminator_learning_rate))
      self.__domain_discriminator.trainable = False
      self.__train_target.compile(loss="binary_crossentropy", optimizer = o.Adam(lr=self.__target_representer_learning_rate))
      #TODO: Possibly not 
      self.__target_model.compile(loss="categorical_crossentropy", optimizer = o.Adam(lr=self.__target_representer_learning_rate), metrics=["accuracy"])
    if self.__algorithm == "wadda":
      self.__domain_discriminator.trainable = True
      self.__domain_discriminator.compile(loss=wasserstein_loss, optimizer = o.Adam(lr=self.__discriminator_learning_rate))
      self.__domain_discriminator.trainable = False
      self.__train_target.compile(loss=wasserstein_loss, optimizer = o.Adam(lr=self.__target_representer_learning_rate))
      #TODO: Possibly not 
      self.__target_model.compile(loss="categorical_crossentropy", optimizer = o.Adam(lr=self.__target_representer_learning_rate), metrics=["accuracy"])


  def __build_domain_discriminator(self, domain_discriminator):
    if self.__algorithm == "adda":
      if domain_discriminator == "linear":
        latent_representation = l.Input(shape=(self.__latent_dimensions,))
        classifier = l.Dense(1, activation="sigmoid")(latent_representation)
        self.__domain_discriminator = Model(latent_representation, classifier)
        self.__domain_discriminator.name = "DomainDiscriminator"
      else:
        assert isinstance(domain_discriminator, K.engine.training.Model), "Provide keras model for domain discriminator"
        assert domain_discriminator.output_shape == self.__nlabels, "Domain discriminator must be a binary classifier"
        assert domain_discriminator.input_shape == self.__latent_dimensions, "Domain discriminator input dimensionality was invalid"
        self.__domain_discriminator = domain_discriminator
        if self.__domain_discriminator.name != "DomainDiscriminator":
          self.__domain_discriminator.name = "DomainDiscriminator"
    elif self.__algorithm == "wadda":
      if domain_discriminator == "linear":
        latent_representation = l.Input(shape=(self.__latent_dimensions,))
        classifier = l.Dense(1, activation = 'linear', kernel_initializer='he_normal',
            W_constraint = WeightClip(self.__weight_clip_threshold))(latent_representation)
        #TODO: Add a WARNING!
        self.__domain_discriminator = Model(latent_representation, classifier)
        self.__domain_discriminator.name = "DomainDiscriminator"
    

  def __train_domain_discriminator(self, iterations, target_label, source_label):
    for _ in range(iterations):
      #TODO:issue Tensorboard   
      target_latent = self.__target_representer.predict(self.target_data.get_batch()[0])
      source_latent = self.__source_representer.predict(self.source_data.get_batch()[0])   
      #TODO:Handle source batch differently?
      self.__domain_discriminator.train_on_batch(target_latent, target_label)
      self.__domain_discriminator.train_on_batch(source_latent, source_label)


  def __initialize_models(self, source_representer, source_classifier, domain_discriminator):
    self.__source_representer = copy_model(source_representer, "SourceRepresenter")
    self.__target_representer = copy_model(source_representer, "TargetRepresenter")
    self.__source_classifier = copy_model(source_classifier, "Classifier")
    self.__build_domain_discriminator(domain_discriminator)


  def __define_models_for_training_and_inference(self):
    representer_input = l.Input(self.input_shape)

    source_representer_output = self.__source_representer(representer_input)
    source_classifier = self.__source_classifier(source_representer_output)
    self.__source_model = Model(representer_input, source_classifier)
    self.__source_model.name = "SourceModel"

    target_representer_output = self.__target_representer(representer_input)
    target_classifier = self.__source_classifier(target_representer_output)
    self.__target_model = Model(representer_input, target_classifier)
    self.__target_model.name = "TargetModel"

    target_representer_output = self.__target_representer(representer_input)
    domain_discriminator_o_target_representer = self.__domain_discriminator(target_representer_output)
    self.__train_target = Model(representer_input, domain_discriminator_o_target_representer)
    self.__train_target.name = "TrainTarget"


  def fit(self, Xtarget, Xsource, iterations=None, validation_data = None, iterations_per_validation = 1, check_mode_collapse = True):
    """
    Xtarget : numpy array of target data or absolute/relative path of the folder, where target data files exist in 

    Xsource : numpy array of source data or absolute/relative path of the folder, where source data files exist in 

    iterations  : integer, number of epochs that model will be trained for

    logs: Boolean, Whether to monitor validation accuracy
    """

    if iterations is None: iterations = self.__epochs

    self.target_data = Xtarget
    self.source_data = Xsource
    source_label = np.ones((self.__batch_size, 1))
    if self.__algorithm == 'adda':
      target_label = np.zeros((self.__batch_size, 1))
    if self.__algorithm == "wadda":
      target_label = -np.ones((self.__batch_size, 1))

    self.__train_domain_discriminator(self.__discriminator_per_representer_iterations_for0, target_label, source_label)
    if validation_data is None:
      for _ in tqdm(range(iterations-1)):
        self.__train_target.train_on_batch(self.target_data.get_batch()[0], source_label)
        self.__train_domain_discriminator(self.__discriminator_per_representer_iterations, target_label, source_label)
      self.__train_target.train_on_batch(self.target_data.get_batch()[0], source_label)  
    else:
      for _iter in tqdm(range(iterations-1)):
        self.__train_target.train_on_batch(self.target_data.get_batch()[0], source_label)
        self.__train_domain_discriminator(self.__discriminator_per_representer_iterations, target_label, source_label)
        if (_iter % iterations_per_validation == 0) and (_iter !=0): 
          print("Iteration: ", _iter, ", accuracy: ", self.evaluate(Xtarget))
          if check_mode_collapse:
            predictions = self.predict(Xtarget)
      self.__train_target.train_on_batch(self.target_data.get_batch()[0], source_label)    
     

  def predict(self,Xtarget):
    """
    Wrapper for target_model.predict()
    """

    return self.target_model.predict(Xtarget)


  def evaluate(self, value, labels=None):
    """
    #TODO: Write description
    """

    if isinstance(value, str):
      assert os.path.exists(value), "Invalid validation set directory"
      validation_data = BatchGenerator(value, -1, self.input_shape,
                            self.__nlabels, self.__shuffle, True)
      X, y = validation_data.get_batch()
      y = one_hot(y,np.array(list(self.__index_to_label_dictionary.keys())))
      return self.target_model.evaluate(X,y)

    if isinstance(value, np.ndarray):
      try:
        #TODO: Possibly review?
        labels = one_hot(labels,np.array(list(self.__index_to_label_dictionary.keys())))
      except Exception:
        raise TypeError("Invalid labels were passed. Must be convertable to string.")
      assert value.shape[1:] == self.__shape[1:], "Invalid target domain dimensions"
      return self.target_model.evaluate(value,labels)
    

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


  @property
  def dpr(self):
    """
    # of iterations that the domain discriminator will be trained on 
    for every iteration the target representer is trained on
    """

    return self.__discriminator_per_representer_iterations

  @dpr.setter
  def dpr(self,value):
    assert value >= 1, "dpr must be >= 1"
    self.__discriminator_per_representer_iterations = value


  @property
  def dpr0(self):
    """
    # of iterations that the domain discriminator will be trained on 
    before adversarial training starts
    """

    return self.__discriminator_per_representer_iterations_for0

  @dpr.setter
  def dpr0(self,value):
    assert value >= 1, "dpr must be >= 1"
    self.__discriminator_per_representer_iterations_for0 = value


  @property
  def domain_discriminator(self):
    return self.__domain_discriminator


  @property
  def target_model(self):
    return self.__target_model


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
      self.__target_data = BatchGenerator(value, self.__batch_size, self.input_shape,
                                          self.__nlabels, self.__shuffle, False)

    if isinstance(value, np.ndarray):
      assert value.shape[1:] == self.__shape[1:], "Invalid target domain dimensions"
      self.__target_data = BatchGenerator_Numpy(value, self.__batch_size, self.__shuffle)


  @property
  def source_data(self):
    return self.__source_data

  @source_data.setter
  def source_data(self, value):
    if isinstance(value, str):
      assert os.path.exists(value), "Invalid source domain directory"
      self.__source_data = BatchGenerator(value,self.__batch_size, self.input_shape, 
                                          self.__nlabels, self.__shuffle,False)

    if isinstance(value, np.ndarray):
      assert value.shape[1:] == self.__shape[1:], "Invalid source domain dimensions"
      self.__source_data = BatchGenerator_Numpy(value, self.__batch_size, self.__shuffle)
