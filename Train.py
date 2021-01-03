import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
import pickle
from tensorflow.keras import datasets, layers, models
from PrepareDataset import prepareDataset
import os

basePath = os.path.dirname(os.path.abspath(__file__))

def startTrain():
  prepareDataset()
  pickle_in = open("XS.pickle", "rb")
  X = pickle.load(pickle_in)

  pickle_in = open("ys.pickle", "rb")
  y = pickle.load(pickle_in)

  X = np.asarray(X).astype('float32')

  y = np.asarray(y).astype('float32')

  model = models.Sequential()
  model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(10))
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  model.fit(tf.convert_to_tensor(X), tf.convert_to_tensor(y), batch_size=15, epochs=10, validation_split=0.1)
  model.save(f'{basePath}/model')
